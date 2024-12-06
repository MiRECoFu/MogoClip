import clip
import os
import torch
import numpy as np
import sys
from os.path import join as pjoin
from torch.utils.data import DataLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.join(current_dir, '..'))
from motion_vae.model import RVQVAE
from options.train_opt import TrainT2MOptions
from utils.get_opt import get_opt
from utils.paramUtil import t2m_kinematic_chain
from data_process.motion_dataset import Text2MotionDataset
from models.mogo_clip import MogoClip
# from motion_loaders.dataset_motion_loader import get_dataset_motion_loader


def load_and_freeze_clip(clip_version):
    clip_model, clip_preprocess = clip.load(clip_version, device='cuda',
                                            jit=False)  # Must set jit=False for training
    # Cannot run on cpu
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16
    # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model

def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, "cuda")
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.vq_name}')
    return vq_model, vq_opt

if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    opt.data_root = '/root/autodl-tmp/HumanML3D'
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.joints_num = 22
    opt.max_motion_len = 55
    opt.text_dir = pjoin(opt.data_root, 'texts')
    dim_pose = 263
    radius = 4
    fps = 20
    kinematic_chain = t2m_kinematic_chain
    dataset_opt_path = '/root/autodl-tmp/checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    clip_version = 'ViT-L/14'
    # checkpoints_dir = '/root/autodl-tmp/checkpoints'
    # dataset_name = 't2m'
    # vq_name = 'rvq_n8192_d128'
    dim_pose = 263
    clip_model = load_and_freeze_clip(clip_version)
    
    vq_model, vq_opt = load_vq_model()
    vq_model.to("cuda")
    vq_model.eval()
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)
    print(f"Train dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mogo_clip = MogoClip(
        embed_dim=768,
        layers=3,
        heads=2,
        width=768*4,
        codebook_size=8192,
        max_motion_length=210
    )
    mogo_clip = mogo_clip.to(device)
    for i, batch_data in enumerate(train_loader):
        captions, motions, m_lens = batch_data
        # print(captions)
        raw_text = captions[0]
        # motion = (motions[0])
        motions = motions.detach().float().to(device)
        text = clip.tokenize(raw_text, truncate=True).to(device)
        # print(f"tokenized: {text} {text.shape}")
        feat_clip_text = clip_model.encode_text(text).float()
        # print(f"embed: {feat_clip_text} {feat_clip_text.shape}")
        code_idx, _motion_emb = vq_model.encode(motions)
        # print(f"code_idx: {code_idx[0]}")
        print(f"code_idx: {code_idx[:, :, 0].shape}")
        encoded_motion = mogo_clip.encode_motion_code(code_idx[:, :, 0])
        print(f"encoded_motion: {encoded_motion} \n shape: {encoded_motion.shape}")
        # print(m_lens)
        if i == 1:
            break
    