import clip
import os
import torch
import numpy as np
import sys
from os.path import join as pjoin
from torch.utils.data import DataLoader
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.join(current_dir, '..'))
from motion_vae.model import RVQVAE
from options.train_opt import TrainT2MOptions
from utils.get_opt import get_opt
from utils.paramUtil import t2m_kinematic_chain
from data_process.motion_dataset import Text2MotionDataset
from models.mogo_clip import MogoClip
from utils.metrics import *
from utils.utils import *

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

def load_mogo_clip():
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
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', 'best_R1.tar'),
                            map_location=device)
    model_key = 'mogo_clip'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = mogo_clip.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading mogo_clip {model_opt.name} from epoch {ckpt["ep"]}!')
    return mogo_clip

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
    dim_pose = 263
    vq_model, vq_opt = load_vq_model()
    vq_model.to("cuda")
    vq_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=device)
    
    mogo_clip = load_mogo_clip()
    mogo_clip.to("cuda")
    mogo_clip.eval()
    
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))
    
    all_file = pjoin(opt.data_root, 'all.txt')
    dataset = Text2MotionDataset(opt, mean, std, all_file)
    
    loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, drop_last=True)
    eval_cosine_pos = []
    eval_cosine_neg = []
    eval_cosine_sep = []
    eval_r1 = []
    eval_r3 = []
    eval_r5 = []
    eval_r10 = []           
    eval_mrr = []
    for i, batch_data in tqdm(enumerate(loader)):
        captions, motions, m_lens = batch_data
        motions = motions.detach().float().to(device)
        code_idx, _motion_emb = vq_model.encode(motions)
        
        motion_code = code_idx[:, :, 0]
        positive_mean, negative_mean, separation = mogo_clip.mean_cosine_similarity(motion_code, captions)
        logits_per_motion, logits_per_text = mogo_clip(motion_code, captions)
        
        eval_cosine_pos.append(positive_mean)
        eval_cosine_neg.append(negative_mean)
        eval_cosine_sep.append(separation)
        logits = logits_per_motion  # 取 logits_per_motion 或 logits_per_text
        # R@1, R@3, R@5, R@10
        r1 = recall_at_k(logits, k=1)
        r3 = recall_at_k(logits, k=3)
        r5 = recall_at_k(logits, k=5)
        r10 = recall_at_k(logits, k=10)
        # Mean Reciprocal Rank
        mrr = mean_reciprocal_rank(logits)
        eval_r1.append(r1)
        eval_r3.append(r3)
        eval_r5.append(r5)
        eval_r10.append(r10)
        eval_mrr.append(mrr)
    
    print(f"Eval eval_cosine_pos:{np.mean(eval_cosine_pos):.3f}, Eval eval_cosine_neg:{np.mean(eval_cosine_neg):.3f}, Eval eval_cosine_sep:{np.mean(eval_cosine_sep):.3f} , R1:{np.mean(eval_r1):.3f}, R3:{np.mean(eval_r3):.3f}, R5:{np.mean(eval_r5):.3f}, R10:{np.mean(eval_r10):.3f}, MRR:{np.mean(eval_mrr):.3f}")

        
    
    
    
    