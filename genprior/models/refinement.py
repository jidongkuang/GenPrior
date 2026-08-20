"""Generative Prototype Refinement for GenPrior."""

from .alignment import CrossModalAlignment
import torch

from genprior.prediction import fuse_distances


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution. Lower = more confident."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class GenPriorModel(CrossModalAlignment):
    """GenPrior with GPR and configurable prediction-space selection."""

    def _gzsl_apply_bias_and_pred(self, distances, seen_classes, unseen_classes):
        """Apply the calibration used by base GZSL inference."""
        seen_class_indices = list(seen_classes)
        unseen_class_indices = list(unseen_classes)
        nearest_seen_distance = torch.min(distances[:, seen_class_indices], dim=1).values
        nearest_unseen_distance = torch.min(distances[:, unseen_class_indices], dim=1).values
        seen_unseen_ratio = nearest_seen_distance / (nearest_unseen_distance + 1e-8)
        predicted_seen_mask = (seen_unseen_ratio < self.args.calibration_factor).unsqueeze(1)

        seen_class_mask = torch.zeros(1, distances.shape[1], dtype=torch.bool, device=distances.device)
        seen_class_mask[:, seen_class_indices] = True
        unseen_class_mask = ~seen_class_mask

        distances = distances.clone()
        distances[predicted_seen_mask & unseen_class_mask] += 1e6
        distances[~predicted_seen_mask & seen_class_mask] += 1e6
        return torch.argmin(distances, dim=1)

    @torch.no_grad()
    def collect_gzsl_gpr_candidates(self, skeleton_features, seen_classes, unseen_classes, class_semantics):
        """Run GZSL GPR pass 1 and return features, assignments, and entropy.

        Prototype assignments are uncalibrated. Public predictions and the
        returned calibration branch use the configured seen/unseen threshold.
        """
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
        reconstructed_semantic = self.semantic_decoder(z_visual)
        reconstructed_visual = self.visual_decoder(z_semantic)

        sample_latent_features = z_visual.mean(1)
        class_latent_prototypes = z_semantic.mean(1)
        latent_distances = torch.cdist(sample_latent_features, class_latent_prototypes, p=2)
        latent_predictions = torch.argmin(latent_distances, dim=1)
        latent_entropy = softmax_entropy(-latent_distances)

        reconstructed_semantic_features = reconstructed_semantic.mean(1)
        class_semantic_prototypes = semantics_all.mean(1)
        semantic_distances = torch.cdist(reconstructed_semantic_features, class_semantic_prototypes, p=2)
        semantic_predictions = torch.argmin(semantic_distances, dim=1)
        semantic_entropy = softmax_entropy(-semantic_distances)

        sample_visual_features = visual_features.mean(1)
        class_visual_prototypes = reconstructed_visual.mean(1)
        visual_distances = torch.cdist(sample_visual_features, class_visual_prototypes, p=2)
        visual_predictions = torch.argmin(visual_distances, dim=1)
        visual_entropy = softmax_entropy(-visual_distances)

        fused_distances = fuse_distances(latent_distances, visual_distances)
        fused_predictions = torch.argmin(fused_distances, dim=1)
        fused_entropy = softmax_entropy(-fused_distances)

        # Expose the branch for the optional historical round-2 diagnostic.
        seen_class_indices = list(seen_classes)
        unseen_class_indices = list(unseen_classes)
        nearest_seen_distance = torch.min(fused_distances[:, seen_class_indices], dim=1).values
        nearest_unseen_distance = torch.min(fused_distances[:, unseen_class_indices], dim=1).values
        seen_unseen_ratio = nearest_seen_distance / (nearest_unseen_distance + 1e-8)
        calib_predicted_seen_mask = seen_unseen_ratio < self.args.calibration_factor

        z_semantic_mean = z_semantic.mean(1)
        reconstructed_visual_mean = reconstructed_visual.mean(1)
        semantic_joint_features = torch.cat([z_semantic_mean, reconstructed_visual_mean], dim=-1)
        z_visual_mean = z_visual.mean(1)
        visual_feature_mean = visual_features.mean(1)
        visual_joint_features = torch.cat([z_visual_mean, visual_feature_mean], dim=-1)
        projected_visual_joint = self.joint_projection(visual_joint_features)
        projected_semantic_joint = self.joint_projection(semantic_joint_features)
        joint_distances = torch.cdist(projected_visual_joint, projected_semantic_joint, p=2)
        joint_predictions = torch.argmin(joint_distances, dim=1)
        joint_entropy = softmax_entropy(-joint_distances)

        prediction_result = self._prediction_result(
            self._gzsl_apply_bias_and_pred(fused_distances, seen_classes, unseen_classes),
            latent_predictions=self._gzsl_apply_bias_and_pred(
                latent_distances, seen_classes, unseen_classes
            ),
            semantic_predictions=self._gzsl_apply_bias_and_pred(
                semantic_distances, seen_classes, unseen_classes
            ),
            visual_predictions=self._gzsl_apply_bias_and_pred(
                visual_distances, seen_classes, unseen_classes
            ),
            joint_predictions=self._gzsl_apply_bias_and_pred(
                joint_distances, seen_classes, unseen_classes
            ),
        )
        candidate_features = {
            'latent': sample_latent_features, 'text': reconstructed_semantic_features, 'visual': sample_visual_features, 'joint': projected_visual_joint,
        }
        assignments = {
            'latent': latent_predictions, 'text': semantic_predictions, 'visual': visual_predictions,
            'fused': fused_predictions, 'joint': joint_predictions,
        }
        entropies = {
            'latent': latent_entropy, 'text': semantic_entropy, 'visual': visual_entropy,
            'fused': fused_entropy, 'joint': joint_entropy,
        }
        return prediction_result, candidate_features, assignments, entropies, calib_predicted_seen_mask

    @torch.no_grad()
    def predict_gzsl_with_prototypes(
        self, skeleton_features, seen_classes, unseen_classes, class_semantics,
        latent_prototypes, semantic_prototypes, visual_prototypes, joint_prototypes
    ):
        """Predict with GPR-refined prototypes and GZSL calibration."""
        visual_features = skeleton_features
        visual_features = visual_features.unsqueeze(1).expand(-1, self.semantic_sequence_length, -1)
        z_visual, _, _ = self.visual_encoder(visual_features)
        reconstructed_semantic = self.semantic_decoder(z_visual)

        sample_latent_features = z_visual.mean(1)
        reconstructed_semantic_features = reconstructed_semantic.mean(1)
        sample_visual_features = visual_features.mean(1)

        latent_distances = torch.cdist(sample_latent_features, latent_prototypes, p=2)
        semantic_distances = torch.cdist(reconstructed_semantic_features, semantic_prototypes, p=2)
        visual_distances = torch.cdist(sample_visual_features, visual_prototypes, p=2)

        fused_distances = fuse_distances(latent_distances, visual_distances)

        z_visual_mean = z_visual.mean(1)
        visual_feature_mean = visual_features.mean(1)
        projected_visual_joint = self.joint_projection(torch.cat([z_visual_mean, visual_feature_mean], dim=-1))
        joint_distances = torch.cdist(projected_visual_joint, joint_prototypes, p=2)

        return self._prediction_result(
            self._gzsl_apply_bias_and_pred(fused_distances, seen_classes, unseen_classes),
            latent_predictions=self._gzsl_apply_bias_and_pred(
                latent_distances, seen_classes, unseen_classes
            ),
            semantic_predictions=self._gzsl_apply_bias_and_pred(
                semantic_distances, seen_classes, unseen_classes
            ),
            visual_predictions=self._gzsl_apply_bias_and_pred(
                visual_distances, seen_classes, unseen_classes
            ),
            joint_predictions=self._gzsl_apply_bias_and_pred(
                joint_distances, seen_classes, unseen_classes
            ),
        )

    @torch.no_grad()
    def collect_zsl_gpr_candidates(self, skeleton_features, unseen_classes, class_semantics):
        """Return pass-1 features, assignments, and entropy for GPR."""
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

        # Latent
        sample_latent_features = z_visual.mean(1)
        class_latent_prototypes = z_semantic.mean(1)
        latent_distances = torch.cdist(sample_latent_features, class_latent_prototypes, p=2)
        latent_predictions = torch.argmin(latent_distances, dim=1)
        latent_entropy = softmax_entropy(-latent_distances)

        reconstructed_semantic = self.semantic_decoder(z_visual)
        reconstructed_visual = self.visual_decoder(z_semantic)
        reconstructed_semantic_features = reconstructed_semantic.mean(1)
        class_semantic_prototypes = semantics_unseen.mean(1)
        semantic_distances = torch.cdist(reconstructed_semantic_features, class_semantic_prototypes, p=2)
        semantic_predictions = torch.argmin(semantic_distances, dim=1)
        semantic_entropy = softmax_entropy(-semantic_distances)

        sample_visual_features = visual_features.mean(1)
        class_visual_prototypes = reconstructed_visual.mean(1)
        visual_distances = torch.cdist(sample_visual_features, class_visual_prototypes, p=2)
        visual_predictions = torch.argmin(visual_distances, dim=1)
        visual_entropy = softmax_entropy(-visual_distances)

        fused_distances = fuse_distances(latent_distances, visual_distances)
        fused_predictions = torch.argmin(fused_distances, dim=1)
        fused_entropy = softmax_entropy(-fused_distances)

        visual_feature_mean = visual_features.mean(1)
        z_visual_mean = z_visual.mean(1)
        visual_joint_features = torch.cat([z_visual_mean, visual_feature_mean], dim=-1)
        projected_visual_joint = self.joint_projection(visual_joint_features)
        z_semantic_mean = z_semantic.mean(1)
        reconstructed_visual_mean = reconstructed_visual.mean(1)
        semantic_joint_features = torch.cat([z_semantic_mean, reconstructed_visual_mean], dim=-1)
        projected_semantic_joint = self.joint_projection(semantic_joint_features)
        joint_distances = torch.cdist(projected_visual_joint, projected_semantic_joint, p=2)
        joint_predictions = torch.argmin(joint_distances, dim=1)
        joint_entropy = softmax_entropy(-joint_distances)

        prediction_result = self._prediction_result(
            fused_predictions,
            latent_predictions=latent_predictions,
            semantic_predictions=semantic_predictions,
            visual_predictions=visual_predictions,
            joint_predictions=joint_predictions,
        )
        candidate_features = {
            'latent': sample_latent_features, 'text': reconstructed_semantic_features, 'visual': sample_visual_features, 'joint': projected_visual_joint,
        }
        assignments = {
            'latent': latent_predictions, 'text': semantic_predictions, 'visual': visual_predictions,
            'fused': fused_predictions, 'joint': joint_predictions,
        }
        entropies = {
            'latent': latent_entropy, 'text': semantic_entropy, 'visual': visual_entropy,
            'fused': fused_entropy, 'joint': joint_entropy,
        }
        return prediction_result, candidate_features, assignments, entropies

    @torch.no_grad()
    def predict_zsl_with_prototypes(
        self, skeleton_features, unseen_classes, class_semantics,
        latent_prototypes, semantic_prototypes, visual_prototypes, joint_prototypes
    ):
        """Predict with the refined ZSL prototypes."""
        visual_features = skeleton_features
        visual_features = visual_features.unsqueeze(1).expand(-1, self.semantic_sequence_length, -1)
        z_visual, _, _ = self.visual_encoder(visual_features)
        reconstructed_semantic = self.semantic_decoder(z_visual)

        sample_latent_features = z_visual.mean(1)
        reconstructed_semantic_features = reconstructed_semantic.mean(1)
        sample_visual_features = visual_features.mean(1)

        latent_distances = torch.cdist(sample_latent_features, latent_prototypes, p=2)
        latent_predictions = torch.argmin(latent_distances, dim=1)

        semantic_distances = torch.cdist(reconstructed_semantic_features, semantic_prototypes, p=2)
        semantic_predictions = torch.argmin(semantic_distances, dim=1)

        visual_distances = torch.cdist(sample_visual_features, visual_prototypes, p=2)
        visual_predictions = torch.argmin(visual_distances, dim=1)

        fused_distances = fuse_distances(latent_distances, visual_distances)
        fused_predictions = torch.argmin(fused_distances, dim=1)

        visual_feature_mean = visual_features.mean(1)
        z_visual_mean = z_visual.mean(1)
        projected_visual_joint = self.joint_projection(torch.cat([z_visual_mean, visual_feature_mean], dim=-1))
        joint_distances = torch.cdist(projected_visual_joint, joint_prototypes, p=2)
        joint_predictions = torch.argmin(joint_distances, dim=1)

        return self._prediction_result(
            fused_predictions,
            latent_predictions=latent_predictions,
            semantic_predictions=semantic_predictions,
            visual_predictions=visual_predictions,
            joint_predictions=joint_predictions,
        )
