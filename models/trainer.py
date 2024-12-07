import torch
import torch.optim as optim
from collections import OrderedDict
from models.mogo_clip import MogoClip
import wandb
import torch.nn.functional as F
from utils.metrics import *
import time
from os.path import join as pjoin
from utils.utils import *


class Trainer:
    def __init__(self, mogo_clip: MogoClip, vq_model, opt, device):
        self.mogo_clip = mogo_clip
        self.vq_model = vq_model
        self.vq_model.eval()
        
        self.lr = opt.lr
        self.max_epoch = opt.max_epoch
        self.opt = opt
        self.device = device
        
        wandb.init(
            # set the wandb project where this run will be logged
            project="MogoClip",

            # track hyperparameters and run metadata
            config={
            "learning_rate": self.lr,
            "epochs": self.max_epoch,
            }
        )
        
    def save(self, file_name, ep, total_it):
        mogo_clip_state_dict = self.mogo_clip.state_dict()
        clip_weights = [e for e in mogo_clip_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del mogo_clip_state_dict[e]
        state = {
            'mogo_clip': mogo_clip_state_dict,
            'opt_mogo_clip': self.opt_mogo_clip.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)
    
        
    def forward(self, batch_data):
        captions, motions, m_lens = batch_data
        # captions = captions.to(self.device)
        motions = motions.detach().float().to(self.device)
        code_idx, _motion_emb = self.vq_model.encode(motions)
        
        motion_code = code_idx[:, :, 0]
        positive_mean, negative_mean, separation = self.mogo_clip.mean_cosine_similarity(motion_code, captions)
        logits_per_motion, logits_per_text = self.mogo_clip(motion_code, captions)        
        batch_size = motions.size(0)      
        labels = torch.arange(batch_size, device=self.device)
        loss_mt = F.cross_entropy(logits_per_motion, labels)
        loss_tm = F.cross_entropy(logits_per_text, labels)
        loss = (loss_mt + loss_tm) / 2
        
        wandb.log({"Train/loss": loss, "Train/positive_cosine": positive_mean, "Train/negative_cosine": negative_mean,"Train/separation": separation})    
        return loss, positive_mean, negative_mean, separation
    
    
    def update(self, batch_data):
        loss, positive_mean, negative_mean, separation = self.forward(batch_data)

        self.opt_mogo_clip.zero_grad()
        loss.backward()
        self.opt_mogo_clip.step()
        # torch.nn.utils.clip_grad_norm_(self.transformotion.parameters(), 0.25)
        self.scheduler.step()

        return loss.item(), positive_mean, negative_mean, separation
    
    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.mogo_clip.load_state_dict(checkpoint['mogo_clip'], strict=False)
        assert len(unexpected_keys) == 0
        # assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_mogo_clip.load_state_dict(checkpoint['opt_mogo_clip']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    
    def train(self, train_loader, val_loader, eval_loader):
        self.mogo_clip.to(self.device)
        self.vq_model.to(self.device)
        
        self.opt_mogo_clip = optim.AdamW(self.mogo_clip.parameters(), lr=self.lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt_mogo_clip, 800000, eta_min=3e-6)
        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))
            
        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        
        
        while epoch < self.max_epoch:
            self.mogo_clip.train()
            self.vq_model.eval()
            
            for i, batch in enumerate(train_loader):
                it += 1
                loss, positive_mean, negative_mean, separation = self.update(batch_data=batch)
                wandb.log({"Train/lr": self.opt_mogo_clip.param_groups[0]['lr']})
                if it % self.opt.log_every == 0:
                    print_current_loss(start_time, it, total_iters, loss, positive_mean, negative_mean, separation, epoch=epoch, inner_iter=i)
                
                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1
            
        
            val_loss = []
            val_positive_cosine_sim = []
            val_neg_cosine_sim = []
            val_cosine_sep = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, positive_mean, negative_mean, separation = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_positive_cosine_sim.append(positive_mean)
                    val_neg_cosine_sim.append(negative_mean)
                    val_cosine_sep.append(separation)
                    

            print(f"Validation loss:{np.mean(val_loss):.3f}, pos_cosine_sim:{np.mean(val_positive_cosine_sim):.3f}, neg_cosine_sim:{np.mean(val_neg_cosine_sim):.3f}, cosine_sep:{np.mean(val_cosine_sep):.3f} ")

            wandb.log({'Val/loss': np.mean(val_loss), 'Val/pos_cosine_sim': np.mean(val_positive_cosine_sim), 'Val/neg_cosine_sim': np.mean(val_neg_cosine_sim), 'Val/cosine_sep': np.mean(val_cosine_sep)})
            
            best_eval_cosine = 0.
            
            if epoch % 10 == 0 or epoch == 1:
                eval_cosine_pos = []
                eval_cosine_neg = []
                eval_cosine_sep = []
                eval_r1 = []
                eval_r3 = []
                eval_r5 = []
                eval_r10 = []           
                eval_mrr = []
                with torch.no_grad():
                    for i, batch_data in enumerate(eval_loader):
                        captions, motions, m_lens = batch_data
                        motions = motions.detach().float().to(self.device)
                        code_idx, _motion_emb = self.vq_model.encode(motions)
                        
                        motion_code = code_idx[:, :, 0]
                        positive_mean, negative_mean, separation = self.mogo_clip.mean_cosine_similarity(motion_code, captions)
                        logits_per_motion, logits_per_text = self.mogo_clip(motion_code, captions)
                        
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

                wandb.log({'Eval/eval_cosine_pos': np.mean(eval_cosine_pos), 'Eval/eval_cosine_neg': np.mean(eval_cosine_neg), 'Eval/eval_cosine_sep': np.mean(eval_cosine_sep), 'Eval/R1':np.mean(eval_r1),'Eval/R3':np.mean(eval_r3), 'Eval/R5':np.mean(eval_r5), 'Eval/R10': np.mean(eval_r10), 'Eval/MRR':np.mean(eval_mrr)})
                if np.mean(eval_cosine) > best_eval_cosine:
                    self.save(pjoin(self.opt.model_dir, 'best_eval_cosine.tar'), epoch, it)
                    best_eval_cosine = np.mean(eval_cosine)
                
                
                