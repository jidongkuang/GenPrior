"""Training and evaluation engine for GenPrior."""
import os
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from genprior.models.refinement import GenPriorModel
from genprior import utils


def _predictions_for(result, head):
    """Read the public fused output or an explicitly requested diagnostic head."""
    if head == 'fused':
        return result['predictions']
    return result['diagnostics'][head]


class GenPriorTrainer:
    def __init__(self, args):
        self.args = args
        self.logger = utils.get_logger(self.args.output_dir)
        self.logger.info(
            f'Training GenPrior on {args.split_name} '
            f'(prediction space: {args.prediction_space}).'
        )

        self.load_task()
        self.load_data()
        self.load_model()
        self.prepare_neighbor_enhanced_semantics()

        self.best_accuracy = [float('-inf')] * 4
        self.best_gpr_accuracy = float('-inf')
        self.best_gpr_method = ''
        self.best_method_name = ''
        self.best_base_accuracy = float('-inf')
        self.best_base_method = ''
        self.best_gzsl_method = ''
        self.best_gzsl_h_base = float('-inf')
        self.best_gzsl_h_gpr = float('-inf')

    def load_task(self):
        self.num_classes, self.unseen_classes, self.seen_classes, self.train_label_dict, self.test_zsl_label_dict, self.test_gzsl_label_dict = utils.task_definition(self.args.split_name)

    def load_data(self):
        feeder_class = utils.import_class(self.args.feeder)
        self.data_loader = dict()
        self.data_loader['train'] = utils.load_data(feeder_class, self.args, utils.init_seed, self.args.train_batch_size, split_type='train')
        self.data_loader['test_zsl'] = utils.load_data(feeder_class, self.args, utils.init_seed, self.args.eval_batch_size, split_type='test_zsl')
        self.data_loader['test_gzsl'] = utils.load_data(feeder_class, self.args, utils.init_seed, self.args.eval_batch_size, split_type='test_gzsl')
        self.train_loader_iter = utils.cycle(self.data_loader['train'])
        self.zsl_loader = self.data_loader['test_zsl']
        self.gzsl_loader = self.data_loader['test_gzsl']

    def load_model(self):
        self.model = GenPriorModel(args=self.args)
        self.model.cuda()
        self.model.train()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

    def _save_checkpoint(self, filename, nb_iter):
        """Save trainable parameters without the frozen text encoder."""
        path = os.path.join(self.args.output_dir, filename)
        full_sd = self.model.state_dict()
        exclude_prefixes = ('text_encoder.',)
        state_dict = {k: v for k, v in full_sd.items() if not k.startswith(exclude_prefixes)}
        extra = {
            'best_zsl': self.best_base_accuracy,
            'best_method': self.best_base_method,
            'prediction_space': self.args.prediction_space,
            'iter': nb_iter,
        }
        if filename == 'best_gpr.pt':
            extra['best_gpr'] = self.best_gpr_accuracy
            extra['best_gpr_method'] = self.best_gpr_method
        if filename == 'best_gzsl.pt':
            extra['best_gzsl_h'] = self.best_accuracy[3]
            extra['best_gzsl_seen'] = self.best_accuracy[1]
            extra['best_gzsl_unseen'] = self.best_accuracy[2]
            extra['best_gzsl_method'] = self.best_gzsl_method
        torch.save({'model': state_dict, **extra}, path)
        self.logger.info(f"Saved checkpoint to {path} ({len(state_dict)} keys, excluded frozen modules)")

    def prepare_neighbor_enhanced_semantics(self):
        if self.args.setting == 'ZSL':
            neighbor_weight = self.args.semantic_neighbor_weights[0]
        elif self.args.setting == 'GZSL':
            neighbor_weight = self.args.semantic_neighbor_weights[1]
        else:
            neighbor_weight = self.args.semantic_neighbor_weights[0]

        base_semantics = self.model.class_semantic_features
        pooled_semantics = base_semantics.mean(1)
        semantic_similarity = F.cosine_similarity(
            pooled_semantics.unsqueeze(1), pooled_semantics.unsqueeze(0), dim=-1
        )
        neighbor_scores, neighbor_indices = torch.topk(
            semantic_similarity,
            k=self.args.semantic_neighbor_count + 1,
            dim=-1,
        )
        neighbor_scores = neighbor_scores[:, 1:]
        neighbor_indices = neighbor_indices[:, 1:]

        self.class_semantics = torch.zeros_like(base_semantics)
        for neighbor_rank in range(self.args.semantic_neighbor_count):
            neighbor_semantics = torch.zeros_like(base_semantics)
            for class_index in range(self.class_semantics.shape[0]):
                neighbor_semantics[class_index] = (
                    neighbor_scores[class_index, neighbor_rank]
                    * base_semantics[
                        neighbor_indices[class_index, neighbor_rank], :, :
                    ]
                ).unsqueeze(0)
            self.class_semantics += (
                neighbor_weight
                / self.args.semantic_neighbor_count
                * neighbor_semantics
            )
        self.class_semantics += base_semantics

    def train(self):
        self.logger.info('Starting cross-modal VAE alignment.')
        avg_loss = 0.0
        avg_mean_alignment = 0.0
        avg_reconstruction = 0.0
        avg_joint_alignment = 0.0

        for nb_iter in range(1, self.args.training_steps + 1):
            skeleton_features, dataset_labels, _ = next(self.train_loader_iter)
            skeleton_features = skeleton_features.cuda().float()
            dataset_labels = dataset_labels.cuda()
            one_hot_labels = torch.tensor(
                [self.train_label_dict[label.item()] for label in dataset_labels]
            ).cuda()

            metrics = self.model.compute_alignment_loss(
                skeleton_features,
                self.class_semantics,
                self.seen_classes,
                torch.argmax(one_hot_labels, dim=1),
            )

            loss = metrics['loss']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()
            avg_mean_alignment += metrics['mean_alignment_loss'].item()
            avg_reconstruction += metrics['reconstruction_loss'].item()
            avg_joint_alignment += metrics['joint_alignment_loss'].item()

            if nb_iter % self.args.log_interval == 0:
                avg_loss /= self.args.log_interval
                avg_mean_alignment /= self.args.log_interval
                avg_reconstruction /= self.args.log_interval
                avg_joint_alignment /= self.args.log_interval
                self.logger.info(
                    f"Iter {nb_iter}: Loss {avg_loss:.4f} "
                    f"(reconstruction={avg_reconstruction:.4f}, "
                    f"mean_alignment={avg_mean_alignment:.4f}, "
                    f"joint_alignment={avg_joint_alignment:.4f})"
                )
                avg_loss = 0.0
                avg_mean_alignment = 0.0
                avg_reconstruction = 0.0
                avg_joint_alignment = 0.0

            if nb_iter % self.args.eval_interval == 0:
                if self.args.setting == 'ZSL':
                    self.test_zsl(nb_iter)
                elif self.args.setting == 'GZSL':
                    self.test_gzsl(nb_iter)
                elif self.args.setting == 'All':
                    self.test_zsl(nb_iter)
                    self.test_gzsl(nb_iter)

        self.print_best_accuracy()

    def test_zsl(self, nb_iter, *, save_checkpoints=True):
        self.model.eval()
        total_samples = 0

        correct_latent = 0
        correct_text = 0
        correct_visual = 0
        correct_fused = 0
        correct_joint = 0

        use_gpr = getattr(self.args, 'use_gpr', True)
        support_ratio = getattr(self.args, 'gpr_support_ratio', 0.9)

        feature_batches = []
        assignment_batches = []
        entropy_batches = []

        for batch_idx, (skeleton_features, label_idx, _) in enumerate(self.zsl_loader):
            with torch.no_grad():
                skeleton_features = skeleton_features.cuda().float()
                true_label_array = torch.tensor([self.test_zsl_label_dict[l.item()] for l in label_idx]).cuda()
                labels = torch.argmax(true_label_array, dim=1)

                if use_gpr:
                    res, feats, preds, ents = self.model.collect_zsl_gpr_candidates(
                        skeleton_features, self.unseen_classes, self.class_semantics
                    )
                    feature_batches.append(feats)
                    assignment_batches.append(preds)
                    entropy_batches.append(ents)
                else:
                    res = self.model.predict_zsl(skeleton_features, self.unseen_classes, self.class_semantics)

                correct_latent += (_predictions_for(res, 'latent') == labels).sum().item()
                correct_text += (_predictions_for(res, 'text') == labels).sum().item()
                correct_visual += (_predictions_for(res, 'visual') == labels).sum().item()
                correct_fused += (res['predictions'] == labels).sum().item()
                correct_joint += (_predictions_for(res, 'joint') == labels).sum().item()

                total_samples += skeleton_features.size(0)

        acc_latent = 100 * correct_latent / total_samples
        acc_text = 100 * correct_text / total_samples
        acc_visual = 100 * correct_visual / total_samples
        acc_fused = 100 * correct_fused / total_samples
        acc_joint = 100 * correct_joint / total_samples

        accs = {'latent': acc_latent, 'text': acc_text, 'visual': acc_visual, 'fused': acc_fused, 'joint': acc_joint}
        prediction_space = self.args.prediction_space
        best_base_current = accs[prediction_space]
        best_base_method = prediction_space
        if best_base_current > self.best_base_accuracy:
            self.best_base_accuracy = best_base_current
            self.best_base_method = best_base_method
            if save_checkpoints:
                self._save_checkpoint('best_base.pt', nb_iter)
        if best_base_current > self.best_accuracy[0]:
            self.best_accuracy[0] = best_base_current
            self.best_method_name = best_base_method
        self.logger.info(f"==> Iter {nb_iter} ZSL Acc (Base):")
        if self.args.diagnostic_heads:
            self.logger.info(f"    Latent: {acc_latent:.2f}%  Text: {acc_text:.2f}%  Visual: {acc_visual:.2f}%  Joint: {acc_joint:.2f}%")
        self.logger.info(
            f"    Predictions ({prediction_space}): {accs[prediction_space]:.2f}%"
        )
        self.logger.info(f"    --> Best Base: {self.best_base_accuracy:.2f}% ({self.best_base_method})")

        if use_gpr and feature_batches:
            feature_pool = {k: torch.cat([f[k] for f in feature_batches]) for k in feature_batches[0]}
            assignment_pool = {k: torch.cat([p[k] for p in assignment_batches]) for k in assignment_batches[0]}
            entropy_pool = {k: torch.cat([e[k] for e in entropy_batches]) for k in entropy_batches[0]}
            n_unseen = len(self.unseen_classes)

            def _estimate_class_prototypes(feature_key, assignment_key, entropy_key, fallback_fn):
                prototypes = []
                features_all = feature_pool[feature_key]
                assignments_all = assignment_pool[assignment_key]
                entropy_all = entropy_pool[entropy_key]
                for i in range(n_unseen):
                    mask = assignments_all == i
                    class_support_set = features_all[mask]
                    class_entropy = entropy_all[mask]
                    support_count = class_entropy.shape[0]
                    if int(support_count * support_ratio) < 1:
                        z = fallback_fn(i)
                    else:
                        num_selected = min(int(support_count * support_ratio), support_count)
                        _, indices = torch.topk(-class_entropy, num_selected)
                        z = torch.mean(class_support_set[indices], dim=0, keepdim=True)
                    prototypes.append(z)
                return torch.cat(prototypes)

            def _fallback_semantic(i):
                semantics_i = self.class_semantics[self.unseen_classes[i:i + 1]]
                if self.args.use_generated_motion_prior:
                    kinematic_prototypes = self.model.generative_kinematic_prototypes[self.unseen_classes[i:i + 1]]
                    intra_class_dispersion = self.model.intra_class_dispersion[self.unseen_classes[i:i + 1]]
                    projected_prior = self.model.generative_prototype_projection(kinematic_prototypes)
                    if hasattr(self.model, 'dispersion_gate'):
                        pooled_semantic_feature = semantics_i.mean(dim=1)
                        fusion_gate_value = self.model.dispersion_gate(pooled_semantic_feature, projected_prior, intra_class_dispersion).unsqueeze(1)
                    else:
                        fusion_gate_value = getattr(self.args, 'prior_fusion_weight', 0.5)
                    projected_prior_exp = projected_prior.unsqueeze(1).expand(-1, self.model.semantic_sequence_length, -1)
                    semantics_i = semantics_i + fusion_gate_value * projected_prior_exp
                z_semantic, _, _ = self.model.semantic_encoder(semantics_i)
                return z_semantic.mean(1)

            semantics_unseen = self.class_semantics[self.unseen_classes]
            if self.args.use_generated_motion_prior:
                kinematic_prototypes = self.model.generative_kinematic_prototypes[self.unseen_classes]
                intra_class_dispersion = self.model.intra_class_dispersion[self.unseen_classes]
                projected_prior = self.model.generative_prototype_projection(kinematic_prototypes)
                if hasattr(self.model, 'dispersion_gate'):
                    fusion_gate_value = self.model.dispersion_gate(semantics_unseen.mean(dim=1), projected_prior, intra_class_dispersion).unsqueeze(1)
                else:
                    fusion_gate_value = getattr(self.args, 'prior_fusion_weight', 0.5)
                projected_prior_exp = projected_prior.unsqueeze(1).expand(-1, self.model.semantic_sequence_length, -1)
                semantics_unseen = semantics_unseen + fusion_gate_value * projected_prior_exp
            z_semantic, _, _ = self.model.semantic_encoder(semantics_unseen)
            reconstructed_visual = self.model.visual_decoder(z_semantic)
            z_semantic_mean = z_semantic.mean(1)
            reconstructed_visual_mean = reconstructed_visual.mean(1)
            semantic_fallback = semantics_unseen.mean(1)
            joint_proto_fallback = self.model.joint_projection(torch.cat([z_semantic_mean, reconstructed_visual_mean], dim=-1))

            latent_protos = _estimate_class_prototypes('latent', 'latent', 'latent', lambda i: _fallback_semantic(i))
            semantic_prototypes = _estimate_class_prototypes('text', 'text', 'text', lambda i: semantic_fallback[i:i + 1])
            visual_protos = _estimate_class_prototypes('visual', 'visual', 'visual', lambda i: reconstructed_visual.mean(1)[i:i + 1])
            joint_protos = _estimate_class_prototypes('joint', 'joint', 'joint', lambda i: joint_proto_fallback[i:i + 1])

            correct_gpr = {k: 0 for k in ['latent', 'text', 'visual', 'fused', 'joint']}
            for batch_idx, (skeleton_features, label_idx, _) in enumerate(self.zsl_loader):
                with torch.no_grad():
                    skeleton_features = skeleton_features.cuda().float()
                    true_label_array = torch.tensor([self.test_zsl_label_dict[l.item()] for l in label_idx]).cuda()
                    labels = torch.argmax(true_label_array, dim=1)
                    gpr_result = self.model.predict_zsl_with_prototypes(
                        skeleton_features, self.unseen_classes, self.class_semantics,
                        latent_protos, semantic_prototypes, visual_protos, joint_protos
                    )
                    for k in correct_gpr:
                        correct_gpr[k] += (_predictions_for(gpr_result, k) == labels).sum().item()

            acc_gpr = {k: 100 * correct_gpr[k] / total_samples for k in correct_gpr}
            best_gpr_current = acc_gpr[prediction_space]
            best_gpr_method = prediction_space
            if best_gpr_current > self.best_gpr_accuracy:
                self.best_gpr_accuracy = best_gpr_current
                self.best_gpr_method = best_gpr_method
                if save_checkpoints:
                    self._save_checkpoint('best_gpr.pt', nb_iter)
            if best_gpr_current > self.best_accuracy[0]:
                self.best_accuracy[0] = best_gpr_current
                self.best_method_name = best_gpr_method
            if self.args.diagnostic_heads:
                self.logger.info(
                    f"    GPR diagnostics: latent={acc_gpr['latent']:.2f}% text={acc_gpr['text']:.2f}% "
                    f"visual={acc_gpr['visual']:.2f}% joint={acc_gpr['joint']:.2f}%"
                )
            self.logger.info(
                f"    GPR predictions ({prediction_space}): "
                f"{acc_gpr[prediction_space]:.2f}%"
            )
            self.logger.info(
                f"    --> Best GPR: {self.best_gpr_accuracy:.2f}% "
                f"({prediction_space})"
            )

        self.model.train()
        metrics = {'base': accs}
        if use_gpr and feature_batches:
            metrics['gpr'] = acc_gpr
        return metrics

    def test_gzsl(self, nb_iter):
        """Evaluate calibrated base and GPR-refined GZSL predictions."""
        self.model.eval()
        use_gpr = getattr(self.args, 'use_gpr', True)
        support_ratio = getattr(self.args, 'gpr_support_ratio', 0.9)
        seen_set = set(self.seen_classes)

        def calc_h_np(all_labels, all_preds):
            all_labels = np.asarray(all_labels).reshape(-1)
            all_preds = np.asarray(all_preds).reshape(-1)
            seen_t = unseen_t = seen_c = unseen_c = 0
            for true_y, pred_y in zip(all_labels, all_preds):
                if true_y in seen_set:
                    seen_t += 1
                    if true_y == pred_y:
                        seen_c += 1
                else:
                    unseen_t += 1
                    if true_y == pred_y:
                        unseen_c += 1
            acc_s = 100 * seen_c / seen_t if seen_t > 0 else 0
            acc_u = 100 * unseen_c / unseen_t if unseen_t > 0 else 0
            h = 2 * acc_s * acc_u / (acc_s + acc_u) if (acc_s + acc_u) > 0 else 0
            return acc_s, acc_u, h

        display_names = {
            'latent': 'Latent  ', 'text': 'Text    ', 'visual': 'Visual  ',
            'fused': 'Fused', 'joint': 'Joint projection',
        }
        keys = ['latent', 'text', 'visual', 'fused', 'joint']

        if not use_gpr:
            stats = {k: {'labels': [], 'preds': []} for k in keys}
            for _, (skeleton_features, label_idx, _) in enumerate(self.gzsl_loader):
                with torch.no_grad():
                    skeleton_features = skeleton_features.cuda().float()
                    true_label_array = torch.tensor([self.test_gzsl_label_dict[l.item()] for l in label_idx]).cuda()
                    labels = torch.argmax(true_label_array, dim=1)
                    res = self.model.predict_gzsl(
                        skeleton_features, self.seen_classes, self.unseen_classes, self.class_semantics
                    )
                    np_labels = labels.cpu().numpy()
                    for k in stats:
                        stats[k]['labels'].append(np_labels)
                        stats[k]['preds'].append(_predictions_for(res, k).cpu().numpy())
            results = {}
            for k in keys:
                s, u, h = calc_h_np(
                    np.concatenate(stats[k]['labels']),
                    np.concatenate(stats[k]['preds']),
                )
                results[k] = (s, u, h)
            self.logger.info(f"==> Iter {nb_iter} GZSL (Base, no GPR) (Seen | Unseen | H):")
            for k in keys:
                s, u, h = results[k]
                if self.args.diagnostic_heads or k == self.args.prediction_space:
                    self.logger.info(f"    {display_names[k]}: {s:.2f} | {u:.2f} | {h:.2f}")
            best_key = self.args.prediction_space
            best_h_now = results[best_key][2]
            if best_h_now > self.best_accuracy[3]:
                self.best_accuracy[1] = results[best_key][0]
                self.best_accuracy[2] = results[best_key][1]
                self.best_accuracy[3] = best_h_now
                self.best_gzsl_method = f'{best_key} (base)'
                self._save_checkpoint('best_gzsl.pt', nb_iter)
            self.logger.info(
                f"    --> Best GZSL H: {self.best_accuracy[3]:.2f}% "
                f"(S:{self.best_accuracy[1]:.2f} U:{self.best_accuracy[2]:.2f}, method={self.best_gzsl_method})"
            )
            self.model.train()
            return

        feature_batches, assignment_batches, entropy_batches = [], [], []
        base_pred_chunks = {k: [] for k in keys}
        label_chunks = []

        for _, (skeleton_features, label_idx, _) in enumerate(self.gzsl_loader):
            with torch.no_grad():
                skeleton_features = skeleton_features.cuda().float()
                true_label_array = torch.tensor([self.test_gzsl_label_dict[l.item()] for l in label_idx]).cuda()
                labels = torch.argmax(true_label_array, dim=1)
                res, feats, preds, ents, _ = self.model.collect_gzsl_gpr_candidates(
                    skeleton_features, self.seen_classes, self.unseen_classes, self.class_semantics
                )
                label_chunks.append(labels.cpu().numpy())
                for k in keys:
                    base_pred_chunks[k].append(_predictions_for(res, k).cpu().numpy())
                feature_batches.append(feats)
                assignment_batches.append(preds)
                entropy_batches.append(ents)

        all_labels = np.concatenate(label_chunks)
        results_base = {}
        for k in keys:
            s, u, h = calc_h_np(all_labels, np.concatenate(base_pred_chunks[k]))
            results_base[k] = (s, u, h)

        self.logger.info(f"==> Iter {nb_iter} GZSL Base (Seen | Unseen | H):")
        for k in keys:
            s, u, h = results_base[k]
            if self.args.diagnostic_heads or k == self.args.prediction_space:
                self.logger.info(f"    {display_names[k]}: {s:.2f} | {u:.2f} | {h:.2f}")
        best_key_base = self.args.prediction_space
        best_h_base = results_base[best_key_base][2]
        self.best_gzsl_h_base = max(self.best_gzsl_h_base, best_h_base)

        num_classes = int(self.class_semantics.shape[0])
        feature_pool = {k: torch.cat([f[k] for f in feature_batches]) for k in feature_batches[0]}
        assignment_pool = {k: torch.cat([p[k] for p in assignment_batches]) for k in assignment_batches[0]}
        entropy_pool = {k: torch.cat([e[k] for e in entropy_batches]) for k in entropy_batches[0]}

        def _estimate_class_prototypes(feature_key, assignment_key, entropy_key, fallback_fn):
            prototypes = []
            features_all = feature_pool[feature_key]
            assignments_all = assignment_pool[assignment_key]
            entropy_all = entropy_pool[entropy_key]
            for c in range(num_classes):
                mask = assignments_all == c
                class_support_set = features_all[mask]
                class_entropy = entropy_all[mask]
                support_count = class_entropy.shape[0]
                if int(support_count * support_ratio) < 1:
                    z = fallback_fn(c)
                else:
                    num_selected = min(int(support_count * support_ratio), support_count)
                    _, indices = torch.topk(-class_entropy, num_selected)
                    z = torch.mean(class_support_set[indices], dim=0, keepdim=True)
                prototypes.append(z)
            return torch.cat(prototypes, dim=0)

        semantics_all = self.class_semantics
        if self.args.use_generated_motion_prior:
            num_cls = semantics_all.shape[0]
            kinematic_prototypes = self.model.generative_kinematic_prototypes[:num_cls]
            intra_class_dispersion = self.model.intra_class_dispersion[:num_cls]
            projected_prior = self.model.generative_prototype_projection(kinematic_prototypes)
            if hasattr(self.model, 'dispersion_gate'):
                fusion_gate_value = self.model.dispersion_gate(semantics_all.mean(dim=1), projected_prior, intra_class_dispersion).unsqueeze(1)
            else:
                fusion_gate_value = getattr(self.args, 'prior_fusion_weight', 0.5)
            projected_prior_exp = projected_prior.unsqueeze(1).expand(-1, self.model.semantic_sequence_length, -1)
            semantics_all = semantics_all + fusion_gate_value * projected_prior_exp
        z_semantic_all, _, _ = self.model.semantic_encoder(semantics_all)
        reconstructed_visual_all = self.model.visual_decoder(z_semantic_all)
        z_semantic_mean_all = z_semantic_all.mean(1)
        reconstructed_visual_mean_all = reconstructed_visual_all.mean(1)
        semantic_prototypes_all = semantics_all.mean(1)
        joint_proto_all = self.model.joint_projection(torch.cat([z_semantic_mean_all, reconstructed_visual_mean_all], dim=-1))

        latent_protos = _estimate_class_prototypes('latent', 'latent', 'latent', lambda c: z_semantic_mean_all[c:c + 1])
        semantic_prototypes = _estimate_class_prototypes('text', 'text', 'text', lambda c: semantic_prototypes_all[c:c + 1])
        visual_protos = _estimate_class_prototypes('visual', 'visual', 'visual', lambda c: reconstructed_visual_mean_all[c:c + 1])
        joint_protos = _estimate_class_prototypes('joint', 'joint', 'joint', lambda c: joint_proto_all[c:c + 1])

        gpr_prediction_chunks = {k: [] for k in keys}
        for _, (skeleton_features, label_idx, _) in enumerate(self.gzsl_loader):
            with torch.no_grad():
                skeleton_features = skeleton_features.cuda().float()
                gpr_result = self.model.predict_gzsl_with_prototypes(
                    skeleton_features, self.seen_classes, self.unseen_classes, self.class_semantics,
                    latent_protos, semantic_prototypes, visual_protos, joint_protos,
                )
                for k in keys:
                    gpr_prediction_chunks[k].append(_predictions_for(gpr_result, k).cpu().numpy())

        results_gpr = {}
        for k in keys:
            s, u, h = calc_h_np(all_labels, np.concatenate(gpr_prediction_chunks[k]))
            results_gpr[k] = (s, u, h)

        self.logger.info("    GPR-refined GZSL (Seen | Unseen | H):")
        for k in keys:
            s, u, h = results_gpr[k]
            if self.args.diagnostic_heads or k == self.args.prediction_space:
                self.logger.info(f"    {display_names[k]}: {s:.2f} | {u:.2f} | {h:.2f}")
        best_key_gpr = self.args.prediction_space
        best_h_gpr = results_gpr[best_key_gpr][2]
        self.best_gzsl_h_gpr = max(self.best_gzsl_h_gpr, best_h_gpr)

        best_h_now = max(best_h_base, best_h_gpr)
        if best_h_base >= best_h_gpr:
            best_src = 'base'
            best_key = best_key_base
            s, u, h = results_base[best_key_base]
        else:
            best_src = 'GPR'
            best_key = best_key_gpr
            s, u, h = results_gpr[best_key_gpr]

        if best_h_now > self.best_accuracy[3]:
            self.best_accuracy[1] = s
            self.best_accuracy[2] = u
            self.best_accuracy[3] = best_h_now
            self.best_gzsl_method = f'{best_key} ({best_src})'
            self._save_checkpoint('best_gzsl.pt', nb_iter)

        self.logger.info(
            f"    --> Best Base H: {best_h_base:.2f}% ({best_key_base}) | "
            f"Best GPR H: {best_h_gpr:.2f}% ({best_key_gpr}) | "
            f"Overall best H: {self.best_accuracy[3]:.2f}% (method={self.best_gzsl_method})"
        )

        self.model.train()

    def print_best_accuracy(self):
        if self.args.setting == 'ZSL':
            self.logger.info(f'Final Best ZSL (Base): {self.best_base_accuracy:.2f}% (method={self.best_base_method})')
            if getattr(self.args, 'use_gpr', True):
                self.logger.info(f'Final Best GPR: {self.best_gpr_accuracy:.2f}%')
                self.logger.info(f'Final Best Overall: {self.best_accuracy[0]:.2f}% (method={self.best_method_name})')
        elif self.args.setting == 'GZSL':
            self.logger.info(
                f'Final Best GZSL - S:{self.best_accuracy[1]:.2f} U:{self.best_accuracy[2]:.2f} '
                f'H:{self.best_accuracy[3]:.2f} (method={self.best_gzsl_method}) | '
                f'hist Base max H:{self.best_gzsl_h_base:.2f} GPR max H:{self.best_gzsl_h_gpr:.2f}'
            )
        elif self.args.setting == 'All':
            self.logger.info(f'Final Best ZSL (Base): {self.best_base_accuracy:.2f}% (method={self.best_base_method})')
            if getattr(self.args, 'use_gpr', True):
                self.logger.info(f'Final Best GPR: {self.best_gpr_accuracy:.2f}%')
                self.logger.info(f'Final Best ZSL Overall: {self.best_accuracy[0]:.2f}% (method={self.best_method_name})')
            self.logger.info(
                f'Final Best GZSL - S:{self.best_accuracy[1]:.2f} U:{self.best_accuracy[2]:.2f} '
                f'H:{self.best_accuracy[3]:.2f} (method={self.best_gzsl_method})'
            )
