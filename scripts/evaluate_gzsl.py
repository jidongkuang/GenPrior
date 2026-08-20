#!/usr/bin/env python3
"""GPR evaluation for generalized zero-shot recognition."""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.filterwarnings('ignore', category=UserWarning)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.getcwd() != str(REPOSITORY_ROOT):
    os.chdir(REPOSITORY_ROOT)
sys.path.insert(0, str(REPOSITORY_ROOT))

from genprior import config, utils
from genprior.trainer import GenPriorTrainer

KEYS = ['latent', 'text', 'visual', 'fused', 'joint']
# Fused predictions combine distances and therefore have no standalone feature tensor.
FEAT_KEYS = ['latent', 'text', 'visual', 'joint']


def predictions_for(result: dict, head: str) -> torch.Tensor:
    if head == 'fused':
        return result['predictions']
    return result['diagnostics'][head]


def calc_h_np(all_labels: np.ndarray, all_preds: np.ndarray, seen_set: set) -> tuple:
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


def collect_all_features_and_labels(loader, test_gzsl_label_dict, device: torch.device):
    feature_chunks, label_chunks = [], []
    for skeleton_features, label_idx, *_ in loader:
        skeleton_features = skeleton_features.to(device=device, dtype=torch.float32, non_blocking=True)
        true_label_array = torch.tensor(
            [test_gzsl_label_dict[l.item()] for l in label_idx],
            device=device,
        )
        encoded_labels = torch.argmax(true_label_array, dim=1)
        feature_chunks.append(skeleton_features.cpu())
        label_chunks.append(encoded_labels.cpu())
    return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)


def run_gzsl_gpr(
    features_cpu: torch.Tensor,
    model,
    args,
    seen_classes,
    unseen_classes,
    class_semantics: torch.Tensor,
    support_ratio: float,
    batch_size: int,
    return_calib_predicted_seen_mask: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], np.ndarray]:
    """Run GPR pass 1, estimate prototypes, and return pass-2 predictions."""
    device = next(model.parameters()).device
    model.eval()
    num_samples = features_cpu.shape[0]
    feature_batches, assignment_batches, entropy_batches = [], [], []
    calib_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_features = features_cpu[i:i + batch_size].to(device)
            _, candidate_features, assignments, entropies, calib_predicted_seen_mask = model.collect_gzsl_gpr_candidates(
                batch_features, seen_classes, unseen_classes, class_semantics
            )
            if return_calib_predicted_seen_mask:
                calib_chunks.append(calib_predicted_seen_mask.detach().cpu().numpy())
            feature_batches.append({k: candidate_features[k].detach() for k in FEAT_KEYS})
            assignment_batches.append({k: assignments[k].detach() for k in KEYS})
            entropy_batches.append({k: entropies[k].detach() for k in KEYS})

    feature_pool = {k: torch.cat([f[k] for f in feature_batches]) for k in FEAT_KEYS}
    assignment_pool = {k: torch.cat([p[k] for p in assignment_batches]) for k in KEYS}
    entropy_pool = {k: torch.cat([e[k] for e in entropy_batches]) for k in KEYS}
    num_classes = int(class_semantics.shape[0])

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

    semantics_all = class_semantics
    if args.use_generated_motion_prior:
        num_cls = semantics_all.shape[0]
        kinematic_prototypes = model.generative_kinematic_prototypes[:num_cls]
        intra_class_dispersion = model.intra_class_dispersion[:num_cls]
        projected_prior = model.generative_prototype_projection(kinematic_prototypes)
        if hasattr(model, 'dispersion_gate'):
            fusion_gate_value = model.dispersion_gate(semantics_all.mean(dim=1), projected_prior, intra_class_dispersion).unsqueeze(1)
        else:
            fusion_gate_value = getattr(args, 'prior_fusion_weight', 0.5)
        projected_prior_exp = projected_prior.unsqueeze(1).expand(-1, args.max_text_length + 1, -1)
        semantics_all = semantics_all + fusion_gate_value * projected_prior_exp
    z_semantic_all, _, _ = model.semantic_encoder(semantics_all)
    reconstructed_visual_all = model.visual_decoder(z_semantic_all)
    z_semantic_mean_all = z_semantic_all.mean(1)
    reconstructed_visual_mean_all = reconstructed_visual_all.mean(1)
    semantic_prototypes_all = semantics_all.mean(1)
    joint_proto_all = model.joint_projection(torch.cat([z_semantic_mean_all, reconstructed_visual_mean_all], dim=-1))

    latent_protos = _estimate_class_prototypes('latent', 'latent', 'latent', lambda c: z_semantic_mean_all[c : c + 1])
    semantic_prototypes = _estimate_class_prototypes('text', 'text', 'text', lambda c: semantic_prototypes_all[c : c + 1])
    visual_protos = _estimate_class_prototypes('visual', 'visual', 'visual', lambda c: reconstructed_visual_mean_all[c : c + 1])
    joint_protos = _estimate_class_prototypes('joint', 'joint', 'joint', lambda c: joint_proto_all[c : c + 1])

    out = {k: [] for k in KEYS}
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_features = features_cpu[i:i + batch_size].to(device)
            gpr_result = model.predict_gzsl_with_prototypes(
                batch_features, seen_classes, unseen_classes, class_semantics,
                latent_protos, semantic_prototypes, visual_protos, joint_protos,
            )
            for k in KEYS:
                out[k].append(predictions_for(gpr_result, k).detach().cpu().numpy())

    preds_out = {k: np.concatenate(out[k]) for k in KEYS}
    if return_calib_predicted_seen_mask:
        return preds_out, np.concatenate(calib_chunks, axis=0)
    return preds_out


_NAMES = {
    'latent': 'Latent  ', 'text': 'Text    ', 'visual': 'Visual  ',
    'fused': 'Fused', 'joint': 'Joint projection',
}


def format_metrics_block(
    title: str,
    results: dict[str, tuple],
    prediction_space: str,
    diagnostics: bool = False,
) -> str:
    """Format the same metrics displayed by :func:`print_block`."""
    lines = [f'=== {title} ===']
    display_keys = KEYS if diagnostics else [prediction_space]
    for k in display_keys:
        s, u, h = results[k]
        lines.append(f"  {_NAMES[k]}: S={s:.2f}% U={u:.2f}% H={h:.2f}%")
    lines.append(f'  prediction_space={prediction_space}')
    return '\n'.join(lines)


def print_block(
    title: str,
    results: dict[str, tuple],
    prediction_space: str,
    diagnostics: bool = False,
):
    print(f'\n=== {title} ===')
    display_keys = KEYS if diagnostics else [prediction_space]
    for k in display_keys:
        s, u, h = results[k]
        print(f"  {_NAMES[k]}: S={s:.2f}% U={u:.2f}% H={h:.2f}%")
    print(f'  prediction_space={prediction_space}')


def append_evaluation_summary(
    path: str,
    *,
    calibration_factor: float,
    gpr_support_ratio: float,
    prior_fusion_weight: float | None,
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    res_r1: dict[str, tuple],
    res_m: dict[str, tuple] | None,
    prediction_space: str,
    diagnostics: bool = False,
) -> None:
    """Append the one-round result and optional historical second-round result."""
    when = datetime.datetime.now().replace(microsecond=0).isoformat()
    block = [
        '',
        f'========== {when} ==========',
        '  1x_GPR = one full-set prototype-refinement round',
        '  2x_GPR = optional calibration-split diagnostic round',
        f'  calibration_factor={calibration_factor}  gpr_support_ratio={gpr_support_ratio}  prior_fusion_weight={prior_fusion_weight}',
        f'  config={config_path}',
        f'  checkpoint={checkpoint_path}',
        f'  output_dir={output_dir}',
        '',
        format_metrics_block(
            '1x GPR (Round1 full-set)', res_r1, prediction_space, diagnostics
        ),
    ]
    if res_m is not None:
        block.extend(
            [
                '',
                format_metrics_block(
                    '2x GPR (Round2 merged)', res_m, prediction_space, diagnostics
                ),
                '',
            ]
        )
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(block))


def metrics_from_preds(labels_np: np.ndarray, assignments: dict[str, np.ndarray], seen_set: set) -> dict[str, tuple]:
    return {k: calc_h_np(labels_np, assignments[k], seen_set) for k in KEYS}


def main():
    p = argparse.ArgumentParser(description='Evaluate GPR in the GZSL setting')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument(
        '--calibration-factor',
        type=float,
        default=None,
        help='override calibration_factor from the YAML config',
    )
    p.add_argument('--prior-fusion-weight', type=float, default=None)
    p.add_argument('--gpr-support-ratio', type=float, default=None)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--output-dir', default='outputs/evaluate_gzsl')
    p.add_argument('--diagnostic-heads', action='store_true')
    p.add_argument(
        '--summary-log',
        type=str,
        default=None,
        help='append one-round and optional diagnostic metrics to this file',
    )
    args_cli, config_rest = p.parse_known_args()

    os.makedirs(args_cli.output_dir, exist_ok=True)
    experiment_parser = config.get_parser()
    with open(args_cli.config, 'r', encoding='utf-8') as f:
        config_values = yaml.safe_load(f) or {}
    config_values = config.normalize_config_values(experiment_parser, config_values)
    experiment_parser.set_defaults(**config_values)
    experiment_parser.set_defaults(setting='GZSL', output_dir=args_cli.output_dir, device=args_cli.device)
    args = experiment_parser.parse_args(['--config', args_cli.config, *config_rest])
    if args_cli.calibration_factor is not None:
        args.calibration_factor = args_cli.calibration_factor
    if args_cli.prior_fusion_weight is not None:
        args.prior_fusion_weight = args_cli.prior_fusion_weight
    if args_cli.gpr_support_ratio is not None:
        args.gpr_support_ratio = args_cli.gpr_support_ratio
    args.setting = 'GZSL'
    args.use_gpr = True
    args.output_dir = args_cli.output_dir

    torch.cuda.set_device(args_cli.device)
    utils.init_seed(args.seed)
    trainer = GenPriorTrainer(args)
    from genprior.checkpoint import load_model_checkpoint
    load_model_checkpoint(trainer.model, args_cli.checkpoint, map_location='cpu')
    trainer.model.cuda()
    model = trainer.model
    device = next(model.parameters()).device

    seen_set = set(trainer.seen_classes)
    bs = args.eval_batch_size
    support_ratio = getattr(args, 'gpr_support_ratio', 0.9)
    class_semantics = trainer.class_semantics

    print(f'calibration_factor={args.calibration_factor}')
    skeleton_features, labels = collect_all_features_and_labels(
        trainer.gzsl_loader, trainer.test_gzsl_label_dict, device
    )
    labels_np = labels.numpy()
    num_samples = labels_np.shape[0]

    # ----- Round 1: full test set -----
    print('\n[Round1] GPR on FULL GZSL test set ...')
    preds_r1, calib_predicted_seen_mask = run_gzsl_gpr(
        skeleton_features, model, args, trainer.seen_classes, trainer.unseen_classes,
        class_semantics, support_ratio, bs, return_calib_predicted_seen_mask=True,
    )
    res_r1 = metrics_from_preds(labels_np, preds_r1, seen_set)
    print_block(
        'Round1 (full-set GPR)',
        res_r1,
        args.prediction_space,
        args_cli.diagnostic_heads,
    )

    if not args.second_refinement_round:
        if args_cli.summary_log:
            append_evaluation_summary(
                args_cli.summary_log,
                calibration_factor=float(args.calibration_factor),
                gpr_support_ratio=float(support_ratio),
                prior_fusion_weight=float(getattr(args, 'prior_fusion_weight', 0.0)),
                config_path=os.path.abspath(args_cli.config),
                checkpoint_path=os.path.abspath(args_cli.checkpoint),
                output_dir=os.path.abspath(args_cli.output_dir),
                res_r1=res_r1,
                res_m=None,
                prediction_space=args.prediction_space,
                diagnostics=args_cli.diagnostic_heads,
            )
        print('\nDone.')
        return

    # ----- Split by calibration (fused_distances seen_unseen_ratio < calibration_factor), no GT -----
    idx_cal_seen = np.where(calib_predicted_seen_mask)[0]
    idx_cal_unseen = np.where(~calib_predicted_seen_mask)[0]
    print(
        f'\n[Split] calib_predicted_seen_mask (seen_unseen_ratio<factor)={len(idx_cal_seen)} '
        f'calib_row_unseen={len(idx_cal_unseen)} total={num_samples}'
    )
    gt_seen_mask = np.isin(labels_np, list(seen_set))
    agree = (calib_predicted_seen_mask == gt_seen_mask).mean()
    print(f'  (diagnostic) calib branch vs GT seen/unseen agreement: {100 * agree:.2f}%')

    # ----- Round 2 per calib subset -----
    preds_r2_cal_seen = {}
    preds_r2_cal_unseen = {}
    if len(idx_cal_seen) > 0:
        print('\n[Round2-calib-seen] GPR on calib_predicted_seen_mask subset ...')
        seen_branch_features = skeleton_features[idx_cal_seen]
        preds_r2_cal_seen = run_gzsl_gpr(
            seen_branch_features, model, args, trainer.seen_classes, trainer.unseen_classes,
            class_semantics, support_ratio, bs,
        )
    if len(idx_cal_unseen) > 0:
        print('\n[Round2-calib-unseen] GPR on ~calib_predicted_seen_mask subset ...')
        unseen_branch_features = skeleton_features[idx_cal_unseen]
        preds_r2_cal_unseen = run_gzsl_gpr(
            unseen_branch_features, model, args, trainer.seen_classes, trainer.unseen_classes,
            class_semantics, support_ratio, bs,
        )

    # ----- Merge by calib branch (same index assignment as Round1 split) -----
    preds_merged = {}
    for k in KEYS:
        merged = np.empty(num_samples, dtype=np.int64)
        if len(idx_cal_seen) > 0:
            merged[idx_cal_seen] = preds_r2_cal_seen[k]
        if len(idx_cal_unseen) > 0:
            merged[idx_cal_unseen] = preds_r2_cal_unseen[k]
        preds_merged[k] = merged

    res_m = metrics_from_preds(labels_np, preds_merged, seen_set)
    print_block(
        'Round2 merged', res_m, args.prediction_space, args_cli.diagnostic_heads
    )

    if args_cli.summary_log:
        append_evaluation_summary(
            args_cli.summary_log,
            calibration_factor=float(args.calibration_factor),
            gpr_support_ratio=float(support_ratio),
            prior_fusion_weight=float(getattr(args, 'prior_fusion_weight', 0.0)),
            config_path=os.path.abspath(args_cli.config),
            checkpoint_path=os.path.abspath(args_cli.checkpoint),
            output_dir=os.path.abspath(args_cli.output_dir),
            res_r1=res_r1,
            res_m=res_m,
            prediction_space=args.prediction_space,
            diagnostics=args_cli.diagnostic_heads,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
