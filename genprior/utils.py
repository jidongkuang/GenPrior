"""Shared utilities for GenPrior experiments."""

import importlib
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F


def get_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f'genprior:{os.path.abspath(output_dir)}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    if logger.handlers:
        return logger

    file_path = os.path.join(output_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

def import_class(import_str):
    module_name, separator, class_name = import_str.rpartition('.')
    if not separator:
        raise ImportError(f'expected a fully qualified class name, got {import_str!r}')
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as error:
        raise ImportError(f'class {class_name!r} was not found in {module_name!r}') from error


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def load_data(Feeder, args, seed_fun, batch_size, split_type='train'):
    data_loader = torch.utils.data.DataLoader(
        dataset=Feeder(args.feature_path, split_type),
        batch_size=batch_size,
        shuffle=True if split_type == 'train' else False,
        num_workers=args.num_workers,
        drop_last=True if split_type == 'train' else False,
        worker_init_fn=seed_fun)
    return data_loader

def process_pkl_to_ntu_format(pkl_data, target_frames=64, target_joints=25):
    N = len(pkl_data)
    # T2M -> NTU Joint Mapping
    joint_remap_indices = [
        0, 3, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21,
        1, 4, 7, 10, 2, 5, 8, 11, 9, 20, 20, 21, 21
    ]
    ntu_data = np.zeros((N, 3, target_frames, target_joints, 2), dtype=np.float32)
    if N == 0: return torch.from_numpy(ntu_data)

    joints_list = [item['joints'] for item in pkl_data]
    max_T = max([j.shape[0] for j in joints_list])
    padded_joints = []
    for j in joints_list:
        pad_len = max_T - j.shape[0]
        if pad_len > 0:
            pad = np.zeros((pad_len, 22, 3))
            j = np.concatenate([j, pad], axis=0)
        padded_joints.append(j)

    joints_np = np.stack(padded_joints, axis=0)
    N, T_src, V_src, C = joints_np.shape
    joints_t = torch.tensor(joints_np, dtype=torch.float)
    joints_t = joints_t.permute(0, 2, 3, 1).contiguous().view(N, V_src * C, T_src)
    joints_t = joints_t.unsqueeze(1)
    joints_interp = F.interpolate(joints_t, size=(V_src * C, target_frames), mode='bilinear', align_corners=False).squeeze(1)
    joints_reshaped = joints_interp.view(N, V_src, C, target_frames).permute(0, 2, 3, 1).contiguous()
    joints_mapped = joints_reshaped[:, :, :, joint_remap_indices]

    # Centralization
    spine_base = joints_mapped[:, :, :, 0:1]
    joints_mapped = joints_mapped - spine_base

    ntu_data[:, :, :, :25, 0] = joints_mapped.cpu().numpy().astype(np.float32)
    return torch.from_numpy(ntu_data) # [N, 3, T, V, 2]

def task_definition(split_name):
    """Return class partitions and label maps for a published split."""
    # split_name: ntu60_xsub_seen55_unseen5 or a named random split
    split_parts = split_name.split('_')
    task_name = '_'.join([split_parts[0], *split_parts[2:]])

    if task_name == 'ntu60_seen55_unseen5':
        num_classes = 60
        unseen_classes = [10, 11, 19, 26, 56]   # ntu60_55/5_split
    elif task_name == 'ntu60_seen48_unseen12':
        num_classes = 60
        unseen_classes = [3,5,9,12,15,40,42,47,51,56,58,59]  # ntu60_48/12_split
    elif task_name == 'ntu120_seen110_unseen10':
        num_classes = 120
        unseen_classes = [4,13,37,43,49,65,88,95,99,106]  # ntu120_110/10_split
    elif task_name == 'ntu120_seen96_unseen24':
        num_classes = 120
        unseen_classes = [5,9,11,16,18,20,22,29,35,39,45,49,59,68,70,81,84,87,93,94,104,113,114,119]  # ntu120_96/24_split
    elif task_name == 'ntu60_seen55_unseen5_sa_dvae_split1':
        num_classes = 60
        unseen_classes = [0, 8, 15, 28, 46]
    elif task_name == 'ntu60_seen55_unseen5_sa_dvae_split2':
        num_classes = 60
        unseen_classes = [15, 19, 23, 47, 50]
    elif task_name == 'ntu60_seen55_unseen5_sa_dvae_split3':
        num_classes = 60
        unseen_classes = [29, 37, 38, 45, 55]
    elif task_name == 'ntu120_seen110_unseen10_sa_dvae_split1':
        num_classes = 120
        unseen_classes = [0, 4, 6, 7, 24, 37, 54, 59, 97, 113]
    elif task_name == 'ntu120_seen110_unseen10_sa_dvae_split2':
        num_classes = 120
        unseen_classes = [63, 79, 86, 92, 98, 100, 103, 110, 111, 117]
    elif task_name == 'ntu120_seen110_unseen10_sa_dvae_split3':
        num_classes = 120
        unseen_classes = [9, 14, 17, 44, 60, 75, 81, 89, 108, 110]
    elif task_name == 'pku51_seen46_unseen5_sa_dvae_split1':
        num_classes = 51
        unseen_classes = [10, 19, 27, 38, 48]
    elif task_name == 'pku51_seen46_unseen5_sa_dvae_split2':
        num_classes = 51
        unseen_classes = [0, 9, 17, 30, 42]
    elif task_name == 'pku51_seen46_unseen5_sa_dvae_split3':
        num_classes = 51
        unseen_classes = [18, 24, 31, 43, 45]
    else:
        raise NotImplementedError('Seen and unseen split errors!')
    seen_classes = list(set(range(num_classes))-set(unseen_classes))  # ntu60
    train_label_dict = {}
    for idx, l in enumerate(seen_classes):
        tmp = [0] * len(seen_classes)
        tmp[idx] = 1
        train_label_dict[l] = tmp
    test_zsl_label_dict = {}
    for idx, l in enumerate(unseen_classes):
        tmp = [0] * len(unseen_classes)
        tmp[idx] = 1
        test_zsl_label_dict[l] = tmp
    test_gzsl_label_dict = {}
    for idx, l in enumerate(range(num_classes)):
        tmp = [0] * num_classes
        tmp[idx] = 1
        test_gzsl_label_dict[l] = tmp
    return num_classes, unseen_classes, seen_classes, train_label_dict, test_zsl_label_dict, test_gzsl_label_dict
