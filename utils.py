# -*- coding: utf-8 -*-


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch):
    lr = init_lr * (0.3 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
