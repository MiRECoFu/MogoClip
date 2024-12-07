import os
import numpy as np
import math
import time


def print_current_loss(start_time, niter_state, total_niters, losses, positive_mean, negative_mean, separation, epoch=None, sub_epoch=None,
                       inner_iter=None, tf_ratio=None, sl_steps=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        print('ep/it:%2d-%4d niter:%6d' % (epoch, inner_iter, niter_state), end=" ")

    message = ' %s completed:%3d%%)' % (time_since(start_time, niter_state / total_niters), niter_state / total_niters * 100)
    # now = time.time()
    # message += '%s'%(as_minutes(now - start_time))


    # for k, v in losses.items():
    message += ' Loss: %.4f ' % losses
    message += ' Cosine positive_mean: %.4f ' % positive_mean
    message += ' Cosine negative_mean: %.4f ' % negative_mean
    message += ' Cosine separation: %.4f ' % separation
    # message += ' sl_length:%2d tf_ratio:%.2f'%(sl_steps, tf_ratio)
    print(message)