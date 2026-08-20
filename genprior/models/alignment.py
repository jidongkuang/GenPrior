import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from genprior import utils
from genprior.prediction import fuse_distances

from .clip_text import load_openai_clip_text
from .shiftgcn.model import Model as ShiftGCNModel

# -----------------------------------------------------------------------------
# Uncertainty-Aware Gating Module
# -----------------------------------------------------------------------------
class DispersionGatedFusion(nn.Module):
    """Fuse generated-motion priors using the configured ablation inputs."""
    def __init__(self, text_feature_dim, dispersion_dim, mode='full'):
        super().__init__()
        if mode not in ('full', 'prototype_dispersion', 'text_prototype'):
            raise ValueError(f'unsupported dispersion gate mode: {mode}')
        self.mode = mode
        if mode == 'text_prototype':
            self.dispersion_projection = None
        else:
            self.dispersion_projection = nn.Linear(
                dispersion_dim, text_feature_dim
            )
        in_dim = text_feature_dim * 3 if mode == 'full' else text_feature_dim * 2
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, semantic_feature, projected_prior, dispersion):
        if self.mode == 'full':
            dispersion_feature = self.dispersion_projection(dispersion)
            gate_input = torch.cat(
                [semantic_feature, projected_prior, dispersion_feature], dim=-1
            )
        elif self.mode == 'prototype_dispersion':
            dispersion_feature = self.dispersion_projection(dispersion)
            gate_input = torch.cat(
                [projected_prior, dispersion_feature], dim=-1
            )
        else:
            gate_input = torch.cat([semantic_feature, projected_prior], dim=-1)
        return self.gate_net(gate_input)

# -----------------------------------------------------------------------------
# Basic VAE Components
# -----------------------------------------------------------------------------
class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
        self.proj_mu = nn.Linear(output_dim, output_dim)
        self.proj_log_var = nn.Linear(output_dim, output_dim)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    def forward(self, x):
        x = self.backbone(x)
        mu = self.proj_mu(x)
        log_var = self.proj_log_var(x)
        latent_sample = self.reparameterize(mu, log_var)
        return [latent_sample, mu, log_var]

class FeatureDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
    def forward(self, x): return self.backbone(x)

# -----------------------------------------------------------------------------
# Cross-modal alignment network
# -----------------------------------------------------------------------------
class CrossModalAlignment(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.load_pretrained_models()
        self.class_semantic_features = self.extract_class_semantic_features()
        # The pooled token adds one position to the encoder sequence.
        self.semantic_sequence_length = int(self.class_semantic_features.shape[1])
        if self.semantic_sequence_length != self.args.max_text_length + 1:
            print(
                f">>> CLIP sequence length is {self.semantic_sequence_length}, not "
                f"max_text_length + 1 ({self.args.max_text_length + 1}); "
                "skeleton features will use the encoder sequence length. <<<"
            )

        if self.args.use_generated_motion_prior:
            print("Initializing generated-motion prior projection...")
            self.generative_prototype_projection = nn.Sequential(
                nn.Linear(args.skeleton_feature_dim, args.text_feature_dim),
                nn.LayerNorm(args.text_feature_dim),
                nn.ReLU()
            )
            gate_mode = getattr(args, 'dispersion_gate_mode', 'full')
            if gate_mode == 'fixed':
                print(">>> Fixed-weight fusion ablation <<<")
            elif getattr(args, 'use_dispersion_gate', True):
                print(f">>> Dispersion-Gated Feature Fusion (mode={gate_mode}) <<<")
                self.dispersion_gate = DispersionGatedFusion(
                    args.text_feature_dim, args.skeleton_feature_dim, mode=gate_mode
                )
            else:
                print(">>> Generated motion prior with fixed fusion weight <<<")

            (
                self.generative_kinematic_prototypes,
                self.intra_class_dispersion,
            ) = self.extract_generated_motion_statistics()

        # Cross-modal variational alignment
        self.semantic_encoder = VariationalEncoder(
            input_dim=args.text_feature_dim, output_dim=args.latent_dim
        )
        self.visual_encoder = VariationalEncoder(
            input_dim=args.skeleton_feature_dim, output_dim=args.latent_dim
        )
        self.visual_decoder = FeatureDecoder(
            input_dim=args.latent_dim, output_dim=args.skeleton_feature_dim
        )
        self.semantic_decoder = FeatureDecoder(
            input_dim=args.latent_dim, output_dim=args.text_feature_dim
        )

        # Compatibility head used by the published checkpoints.
        self.joint_projection = nn.Sequential(
            nn.Linear(args.latent_dim + args.skeleton_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    @staticmethod
    def _prediction_result(
        predictions,
        *,
        latent_predictions,
        semantic_predictions,
        visual_predictions,
        joint_predictions,
    ):
        """Build fused predictions and the other selectable prediction spaces."""
        return {
            'predictions': predictions,
            'diagnostics': {
                'latent': latent_predictions,
                'text': semantic_predictions,
                'visual': visual_predictions,
                'joint': joint_predictions,
            },
        }

    def load_pretrained_models(self):
        backend = getattr(self.args, "clip_backend", "hf")
        if backend == "openai":
            dl = getattr(self.args, "clip_download_root", None)
            self.text_encoder = load_openai_clip_text(
                self.args.text_encoder_name_or_path, device="cuda", download_root=dl
            )
            self.text_encoder = self.text_encoder.to("cuda")
            self.text_encoder.requires_grad_(False)
            self.tokenizer = None
            print(f">>> CLIP text backend: openai (clip.load: {self.args.text_encoder_name_or_path}) <<<")
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.args.text_encoder_name_or_path)
            self.text_encoder = CLIPTextModel.from_pretrained(self.args.text_encoder_name_or_path)
            self.text_encoder = self.text_encoder.to("cuda")
            self.text_encoder.requires_grad_(False)

        if self.args.use_generated_motion_prior:
            print(
                "Loading the skeleton encoder for generated motions from "
                f"{self.args.skeleton_encoder_checkpoint}..."
            )
            self.skeleton_encoder = ShiftGCNModel(num_class=self.args.num_seen_classes, num_point=self.args.num_point, num_person=self.args.num_person, graph=self.args.graph, graph_args=self.args.graph_args).to('cuda')
            checkpoint = torch.load(self.args.skeleton_encoder_checkpoint, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'fc' not in k}
            self.skeleton_encoder.load_state_dict(new_state_dict, strict=False)
            self.skeleton_encoder.requires_grad_(False)
            self.skeleton_encoder.eval()

    def extract_class_semantic_features(self):
        dataset_name = self.args.split_name.split('_')[0]
        def deal_prompts(prompts):
            text_embed = []
            backend = getattr(self.args, "clip_backend", "hf")
            for prompt in prompts:
                if backend == "openai":
                    import clip as openai_clip

                    tokens = openai_clip.tokenize([prompt], truncate=True).to("cuda")
                    embeds = self.text_encoder(tokens)
                else:
                    text_inputs = self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.args.max_text_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to("cuda")
                    embeds = self.text_encoder(text_inputs.input_ids)
                prompt_embeds = embeds["last_hidden_state"]
                pooled_prompt_embeds = embeds["pooler_output"]
                text_embed.append(
                    torch.cat((prompt_embeds, pooled_prompt_embeds.unsqueeze(1)), dim=1).detach().clone()
                )
            return text_embed
        csv_file = os.path.join(self.args.semantic_dir, '{}.csv'.format(dataset_name))
        csv_prompts = pd.read_csv(csv_file)['label'].values.tolist()
        text_embed_csv = deal_prompts(csv_prompts)
        txt_file = os.path.join(self.args.semantic_dir, '{}_llm.txt'.format(dataset_name))
        with open(txt_file, 'r', encoding='utf-8') as file:
            content_list = file.readlines()
        txt_prompts =[line.strip() for line in content_list]
        text_embed_txt = deal_prompts(txt_prompts)
        self.class_semantic_features = torch.cat((torch.cat(text_embed_csv, dim=0), torch.cat(text_embed_txt, dim=0)), dim=-1)
        return self.class_semantic_features

    def extract_generated_motion_statistics(self):
        generated_motion_path = self.args.generated_motion_path
        with open(generated_motion_path, 'rb') as f:
            generated_samples = pickle.load(f)
        dataset_name = self.args.split_name.split('_')[0]
        max_label = 59 if 'ntu60' in dataset_name else 119

        valid_samples = [
            sample
            for sample in generated_samples
            if 0 <= sample.get('label_index', -1) <= max_label
        ]
        generated_motion_tensor = utils.process_pkl_to_ntu_format(valid_samples).float()
        labels = np.array([sample['label_index'] for sample in valid_samples])

        feature_batches = []
        batch_size = 64
        with torch.no_grad():
            for i in tqdm(range(0, len(generated_motion_tensor), batch_size)):
                batch = generated_motion_tensor[i:i+batch_size].to('cuda')
                feat = self.skeleton_encoder(batch)
                feat = feat.mean(3).mean(2)
                feature_batches.append(feat)
        all_features = torch.cat(feature_batches, dim=0)

        num_classes = max_label + 1
        prototypes = torch.zeros(num_classes, 256).to('cuda')
        dispersion = torch.zeros(num_classes, 256).to('cuda')

        label_tensor = torch.tensor(labels).to('cuda')
        for c in range(num_classes):
            class_mask = label_tensor == c
            if class_mask.sum() > 1:
                class_features = all_features[class_mask]
                prototypes[c] = class_features.mean(dim=0)
                dispersion[c] = class_features.std(dim=0)
            elif class_mask.sum() == 1:
                prototypes[c] = all_features[class_mask].mean(dim=0)
                dispersion[c] = torch.zeros_like(prototypes[c])

        del self.skeleton_encoder
        torch.cuda.empty_cache()
        return prototypes.detach(), dispersion.detach()

    def prepare_alignment_features(self, skeleton_features, seen_classes, label_idx, class_semantics):
        visual_features = skeleton_features
        visual_features = visual_features.unsqueeze(1).expand(-1, self.semantic_sequence_length, -1)

        semantics_seen = class_semantics[seen_classes]
        semantic_features = torch.cat([semantics_seen[i,:,:].unsqueeze(0) for i in label_idx], dim=0)

        if self.args.use_generated_motion_prior:
            real_label_ids = [seen_classes[i] for i in label_idx]

            batch_kinematic_prototypes = self.generative_kinematic_prototypes[real_label_ids]
            batch_intra_class_dispersion = self.intra_class_dispersion[real_label_ids]

            projected_prior = self.generative_prototype_projection(batch_kinematic_prototypes)

            if hasattr(self, 'dispersion_gate'):
                pooled_semantic_feature = semantic_features.mean(dim=1)
                fusion_gate_value = self.dispersion_gate(pooled_semantic_feature, projected_prior, batch_intra_class_dispersion)
                fusion_gate_value = fusion_gate_value.unsqueeze(1)
            else:
                fusion_gate_value = getattr(self.args, 'prior_fusion_weight', 0.5)

            projected_prior_expanded = projected_prior.unsqueeze(1).expand(-1, self.semantic_sequence_length, -1)
            semantic_features = semantic_features + fusion_gate_value * projected_prior_expanded

        return visual_features, semantic_features

    def compute_alignment_loss(
        self, skeleton_features, class_semantics, seen_classes, label_idx
    ):
        visual_features, semantic_features = self.prepare_alignment_features(skeleton_features, seen_classes, label_idx, class_semantics)

        # Encoder passes
        z_semantic, mu_semantic, log_variance_semantic = self.semantic_encoder(semantic_features)
        z_visual, mu_visual, log_var_visual = self.visual_encoder(visual_features)

        mean_alignment_loss = F.mse_loss(
            mu_semantic, mu_visual, reduction='mean'
        )
        variance_alignment_loss = F.mse_loss(
            log_variance_semantic, log_var_visual, reduction='mean'
        )

        reconstructed_visual_from_visual = self.visual_decoder(z_visual)
        reconstructed_semantic_from_semantic = self.semantic_decoder(z_semantic)
        reconstructed_semantic_from_visual = self.semantic_decoder(z_visual)
        reconstructed_visual_from_semantic = self.visual_decoder(z_semantic)

        reconstruction_visual_to_visual = F.mse_loss(reconstructed_visual_from_visual, visual_features, reduction='mean')
        reconstruction_semantic_to_semantic = F.mse_loss(reconstructed_semantic_from_semantic, semantic_features, reduction='mean')
        reconstruction_visual_to_semantic = F.mse_loss(reconstructed_semantic_from_visual, semantic_features, reduction='mean')
        reconstruction_semantic_to_visual = F.mse_loss(reconstructed_visual_from_semantic, visual_features, reduction='mean')
        reconstruction_loss = (
            reconstruction_visual_to_visual
            + reconstruction_semantic_to_semantic
            + reconstruction_visual_to_semantic
            + reconstruction_semantic_to_visual
        ) / 4

        z_semantic_mean = z_semantic.mean(1)
        z_visual_mean = z_visual.mean(1)
        visual_feature_mean = visual_features.mean(1)
        reconstructed_visual_semantic_mean = reconstructed_visual_from_semantic.mean(1)

        semantic_joint_features = torch.cat(
            [z_semantic_mean, reconstructed_visual_semantic_mean], dim=-1
        )
        visual_joint_features = torch.cat([z_visual_mean, visual_feature_mean], dim=-1)

        projected_semantic_joint = self.joint_projection(semantic_joint_features)
        projected_visual_joint = self.joint_projection(visual_joint_features)

        joint_alignment_loss = F.mse_loss(
            projected_semantic_joint, projected_visual_joint, reduction='mean'
        )

        total_loss = (
            reconstruction_loss
            + self.args.distribution_alignment_weight
            * (mean_alignment_loss + variance_alignment_loss)
            + joint_alignment_loss
        )

        return {
            'loss': total_loss,
            'mean_alignment_loss': mean_alignment_loss,
            'reconstruction_loss': reconstruction_loss,
            'joint_alignment_loss': joint_alignment_loss,
        }

    @torch.no_grad()
    def predict_zsl(self, skeleton_features, unseen_classes, class_semantics):
        visual_features = skeleton_features
        visual_features = visual_features.unsqueeze(1).expand(-1, self.semantic_sequence_length, -1)
        z_visual, _, _ = self.visual_encoder(visual_features)

        semantics_unseen = class_semantics[unseen_classes]
        if self.args.use_generated_motion_prior:
            kinematic_prototypes = self.generative_kinematic_prototypes[unseen_classes]
            intra_class_dispersion = self.intra_class_dispersion[unseen_classes]
            projected_prior = self.generative_prototype_projection(kinematic_prototypes)

            if hasattr(self, 'dispersion_gate'):
                pooled_semantic_feature = semantics_unseen.mean(dim=1)
                fusion_gate_value = self.dispersion_gate(pooled_semantic_feature, projected_prior, intra_class_dispersion).unsqueeze(1)
            else:
                fusion_gate_value = getattr(self.args, 'prior_fusion_weight', 0.5)

            projected_prior_exp = projected_prior.unsqueeze(1).expand(-1, self.semantic_sequence_length, -1)
            semantics_unseen = semantics_unseen + fusion_gate_value * projected_prior_exp

        z_semantic, _, _ = self.semantic_encoder(semantics_unseen)

        # A. Latent
        sample_latent_features = z_visual.mean(1)
        class_latent_prototypes = z_semantic.mean(1)
        latent_distances = torch.cdist(sample_latent_features, class_latent_prototypes, p=2)
        latent_predictions = torch.argmin(latent_distances, dim=1)

        # B. Text
        reconstructed_semantic = self.semantic_decoder(z_visual)
        reconstructed_semantic_features = reconstructed_semantic.mean(1)
        class_semantic_prototypes = semantics_unseen.mean(1)
        semantic_distances = torch.cdist(reconstructed_semantic_features, class_semantic_prototypes, p=2)
        semantic_predictions = torch.argmin(semantic_distances, dim=1)

        # C. Visual
        reconstructed_visual = self.visual_decoder(z_semantic)
        sample_visual_features = visual_features.mean(1)
        class_visual_prototypes = reconstructed_visual.mean(1)
        visual_distances = torch.cdist(sample_visual_features, class_visual_prototypes, p=2)
        visual_predictions = torch.argmin(visual_distances, dim=1)

        # D. Fused
        fused_distances = fuse_distances(latent_distances, visual_distances)
        fused_predictions = torch.argmin(fused_distances, dim=1)

        # E. Joint projection
        visual_feature_mean = visual_features.mean(1)
        z_visual_mean = z_visual.mean(1)
        visual_joint_features = torch.cat([z_visual_mean, visual_feature_mean], dim=-1)
        projected_visual_joint = self.joint_projection(visual_joint_features)

        z_semantic_mean = z_semantic.mean(1)
        reconstructed_visual_mean = reconstructed_visual.mean(1)
        semantic_joint_features = torch.cat(
            [z_semantic_mean, reconstructed_visual_mean], dim=-1
        )
        projected_semantic_joint = self.joint_projection(semantic_joint_features)

        joint_distances = torch.cdist(
            projected_visual_joint, projected_semantic_joint, p=2
        )
        joint_predictions = torch.argmin(joint_distances, dim=1)

        return self._prediction_result(
            fused_predictions,
            latent_predictions=latent_predictions,
            semantic_predictions=semantic_predictions,
            visual_predictions=visual_predictions,
            joint_predictions=joint_predictions,
        )

    @torch.no_grad()
    def predict_gzsl(self, skeleton_features, seen_classes, unseen_classes, class_semantics):
        visual_features = skeleton_features
        visual_features = visual_features.unsqueeze(1).expand(-1, self.semantic_sequence_length, -1)
        z_visual, _, _ = self.visual_encoder(visual_features)

        semantics_all = class_semantics
        if self.args.use_generated_motion_prior:
            num_cls = semantics_all.shape[0]
            kinematic_prototypes = self.generative_kinematic_prototypes[:num_cls]
            intra_class_dispersion = self.intra_class_dispersion[:num_cls]
            projected_prior = self.generative_prototype_projection(kinematic_prototypes)

            if hasattr(self, 'dispersion_gate'):
                pooled_semantic_feature = semantics_all.mean(dim=1)
                fusion_gate_value = self.dispersion_gate(pooled_semantic_feature, projected_prior, intra_class_dispersion).unsqueeze(1)
            else:
                fusion_gate_value = getattr(self.args, 'prior_fusion_weight', 0.5)

            projected_prior_exp = projected_prior.unsqueeze(1).expand(-1, self.semantic_sequence_length, -1)
            semantics_all = semantics_all + fusion_gate_value * projected_prior_exp

        z_semantic, _, _ = self.semantic_encoder(semantics_all)

        def apply_bias_and_pred(dist_mat):
            dist_seen = torch.min(dist_mat[:, seen_classes], dim=1).values
            dist_unseen = torch.min(dist_mat[:, unseen_classes], dim=1).values
            ratio = dist_seen / (dist_unseen + 1e-8)
            row_seen = (ratio < self.args.calibration_factor).unsqueeze(1)

            col_seen = torch.zeros(1, dist_mat.shape[1], dtype=torch.bool, device=dist_mat.device)
            col_seen[:, seen_classes] = True
            col_unseen = ~col_seen

            dist_mat[row_seen & col_unseen] += 1e6
            dist_mat[~row_seen & col_seen] += 1e6
            return torch.argmin(dist_mat, dim=1)

        sample_latent_features = z_visual.mean(1)
        class_latent_prototypes = z_semantic.mean(1)
        latent_distances = torch.cdist(
            sample_latent_features, class_latent_prototypes, p=2
        )
        latent_predictions = apply_bias_and_pred(latent_distances.clone())

        reconstructed_semantic = self.semantic_decoder(z_visual)
        class_semantic_prototypes = semantics_all.mean(1)
        semantic_distances = torch.cdist(
            reconstructed_semantic.mean(1), class_semantic_prototypes, p=2
        )
        semantic_predictions = apply_bias_and_pred(semantic_distances.clone())

        reconstructed_visual = self.visual_decoder(z_semantic)
        sample_visual_features = visual_features.mean(1)
        visual_distances = torch.cdist(
            sample_visual_features, reconstructed_visual.mean(1), p=2
        )
        visual_predictions = apply_bias_and_pred(visual_distances.clone())

        fused_distances = fuse_distances(latent_distances, visual_distances)
        fused_predictions = apply_bias_and_pred(fused_distances.clone())

        projected_visual_joint = self.joint_projection(
            torch.cat([sample_latent_features, sample_visual_features], dim=-1)
        )
        projected_semantic_joint = self.joint_projection(
            torch.cat(
                [class_latent_prototypes, reconstructed_visual.mean(1)], dim=-1
            )
        )
        joint_distances = torch.cdist(
            projected_visual_joint, projected_semantic_joint, p=2
        )
        joint_predictions = apply_bias_and_pred(joint_distances.clone())

        return self._prediction_result(
            fused_predictions,
            latent_predictions=latent_predictions,
            semantic_predictions=semantic_predictions,
            visual_predictions=visual_predictions,
            joint_predictions=joint_predictions,
        )
