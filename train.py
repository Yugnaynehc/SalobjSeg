# -*- coding: utf-8 -*-

'''
Training code
'''
from __future__ import print_function
from __future__ import absolute_import

import torch
from torch.autograd import Variable
from tensorboard_logger import configure, log_value

from data import get_loader
from model import Model
from args import image_root, gt_root
from args import opt, log_environment
from utils import clip_gradient, adjust_lr


configure(log_environment, flush_secs=10)


# build models
model = Model()

if opt.cuda:
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, num_workers=2, pin_memory=True)
else:
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, num_workers=2, pin_memory=False)

total_step = len(train_loader)

params = model.parameters()
optimizer = torch.optim.Adam(params, lr=opt.lr)
crit = torch.nn.MSELoss()

print("Let's go!")
for epoch in range(1, opt.epoch + 1):
    model.train()
    adjust_lr(optimizer, opt.lr, epoch)
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # Load data
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        if opt.cuda:
            images = images.cuda()
            gts = gts.cuda()
        # Forward
        res = model(images)
        # Merge losses
        loss = crit(res, gts)
        # Backward and update
        loss.backward()
        clip_gradient(optimizer, clip)
        optimizer.step()
        if i % 10 == 0 or i == total_step:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.8f' %
                  (opt.epoch, opt.epoch, i, total_step, loss.data[0]))
