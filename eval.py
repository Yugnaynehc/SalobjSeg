# -*- coding: utf-8 -*-

'''
Evaluation code
'''
from __future__ import print_function
from __future__ import absolute_import

import os
import time
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from data import get_loader
from model import Model
from args import image_root, gt_root, visual_dir
from args import opt, weight_pth_path


def evaluate(model, eval_loader):
    total_step = len(eval_loader)

    print("Let's go!")
    for i, pack in enumerate(eval_loader, start=1):
        # Load data
        images, gts = pack
        images = Variable(images, volatile=True)
        gts = Variable(gts, volatile=True)
        if opt.cuda:
            images = images.cuda()
            gts = gts.cuda()
        # Forward
        t0 = time.time()
        res = model(images)
        times = time.time() - t0
        print(times)
        print('Step [%d/%d], FPS: %.2f' % (i, total_step, len(res) / times))
        save_image(images.data, os.path.join(visual_dir, 'images_test_%d.png' % (i)))
        save_image(gts.data, os.path.join(visual_dir, 'gts_test_%d.png' % (i)))
        save_image(res.data, os.path.join(visual_dir, 'res_test_%d.png' % (i)))


if __name__ == '__main__':
    # build models
    model = Model()
    load_path = weight_pth_path + '.%d' % opt.eval_epoch
    if opt.cuda:
        eval_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, num_workers=2, pin_memory=True)
        weights = torch.load(load_path)
        model.cuda()
    else:
        eval_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, num_workers=2, pin_memory=False)
        weights = torch.load(load_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(weights)
    model.eval()
    evaluate(model, eval_loader)
