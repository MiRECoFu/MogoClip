import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default="trm_xl_b48_d1024_822_clip_nh16_nl18_n8192_downt_1_once", help='Name of this trial')

        self.parser.add_argument('--vq_name', type=str, default="rvq_n8192_d128", help='Name of the rvq model.')

        self.parser.add_argument("--gpu_id", type=int, default=-1, help='GPU id')
        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name, {t2m} for humanml3d, {kit} for kit-ml')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/root/autodl-tmp/checkpoints', help='models are saved here.')

        self.parser.add_argument('--latent_dim', type=int, default=1024, help='Dimension of transformer latent.')
        self.parser.add_argument('--n_heads', type=int, default=16, help='Number of heads.')
        self.parser.add_argument('--n_layers', type=int, default=18, help='Number of attention layers.')
        self.parser.add_argument('--ff_size', type=int, default=1024, help='FF_Size')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='Dropout ratio in transformer')

        self.parser.add_argument("--max_motion_length", type=int, default=210, help="Max length of motion")
        self.parser.add_argument("--unit_length", type=int, default=1, help="Downscale ratio of VQ")

        self.parser.add_argument('--force_mask', action="store_true", help='True: mask out conditions')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.is_train = self.is_train

        if self.opt.gpu_id != -1:
            # self.opt.gpu_id = int(self.opt.gpu_id)
            torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        if self.is_train:
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt