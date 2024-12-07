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
from models.trainer import Trainer


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
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/t2m/', opt.dataset_name, opt.name)
    os.makedirs(opt.model_dir, exist_ok=True)
    # os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    dim_pose = 263
    radius = 4
    fps = 20
    kinematic_chain = t2m_kinematic_chain
    dataset_opt_path = '/root/autodl-tmp/checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    clip_version = 'ViT-L/14'
    dim_pose = 263
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(opt)
    vq_model, vq_opt = load_vq_model()
    vq_model.to("cuda")
    vq_model.eval()
    print(vq_opt)
    mogo_clip = MogoClip(
        embed_dim=opt.embed_dim,
        layers=opt.layers,
        heads=opt.heads,
        width=opt.width,
        codebook_size=vq_opt.nb_code,
        max_motion_length=opt.max_motion_length,
        clip_version=clip_version,
        device=device
    )
    
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')
    eval_split_file = pjoin(opt.data_root, 'test.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)
    eval_dataset = Text2MotionDataset(opt, mean, std, eval_split_file)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    trainer = Trainer(mogo_clip, vq_model, opt, device)
    trainer.train(train_loader, val_loader, eval_loader)