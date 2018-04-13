# -*- coding: utf-8 -*-

import os
import time
import argparse


def prepare_dir(root, ds, trial_id=None):
    if not os.path.exists(root):
        os.mkdir(root)
    directory = os.path.join(root, ds.upper())
    if not os.path.exists(directory):
        os.mkdir(directory)
    if trial_id:
        directory = os.path.join(directory, str(trial_id))
        if not os.path.exists(directory):
            os.mkdir(directory)
    return directory


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=256, help='training batch size')
parser.add_argument('--trainsize', type=int, default=None, help='training dataset size')
parser.add_argument('--valsize', type=int, default=None, help='validation dataset size')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda or not')
parser.add_argument('--nocuda', dest='cuda', action='store_false')
parser.add_argument('--single', type=bool, default=True, help='single GPU')
parser.add_argument('--nosingle', dest='single', action='store_false')
parser.add_argument('--checkpoint', type=bool, default=False, help='use the best checkpoint or not')
parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping margin')
parser.add_argument('--ds', type=str, default='msra10k', help='dataset name')
parser.add_argument('--eval-ds', type=str, default='msra10k', help='eval dataset name')
parser.add_argument('--tb-dir', type=str, default='logs', help='tensorboard logs dir')
parser.add_argument('--result-root', type=str, default='results', help='checkpoint save root')
parser.add_argument('--visual-root', type=str, default='visuals', help='visual result save root')
parser.add_argument('--trial-id', type=str, default=None, help='trial id')
parser.add_argument('--eval-epoch', type=int, default=1, help='evaluate saved model selected by epoch number')
opt = parser.parse_args()


# 训练相关的超参数
ds = opt.ds
trial_id = opt.trial_id

# 模型相关的超参数
frame_shape = (3, 224, 224)                       # 视频帧的形状
feature_size = 2048                               # 最终特征大小，只取表观特征
max_words = 30                                    # 文本序列的最大长度

img_projected_size = 512                          # 图像投影空间的大小
word_projected_size = 512                         # 文本投影空间的大小
hidden_size = 1024                                # 循环网络的隐层单元数目

# 训练日志信息
time_format = '%m-%d_%X'
current_time = time.strftime(time_format, time.localtime())
env_tag = '_'.join(map(str, [ds.upper(), trial_id, current_time, opt.batchsize, opt.lr]))
log_environment = os.path.join(opt.tb_dir, env_tag)   # tensorboard的记录环境


# 数据相关的参数
# 提供两个数据集：MSRA10K, PASCAL, DUT-OMRON
msra10k_image_root = './datasets/MSRA10K/images/'
msra10k_gt_root = './datasets/MSRA10K/gts/'
pascal_image_root = './datasets/PASCAL/images/'
pascal_gt_root = './datasets/PASCAL/gts/'

dataset = {
    'msra10k': [msra10k_image_root, msra10k_gt_root],
    'pascal': [pascal_image_root, pascal_gt_root]
}

image_root, gt_root = dataset[ds]
eval_image_root, eval_gt_root = dataset[opt.eval_ds]

feat_root = 'feats'
feat_dir = prepare_dir(feat_root, ds)

vocab_pkl_path = os.path.join(feat_dir, ds + '_vocab.pkl')

sentence_pkl_base = os.path.join(feat_dir, ds + '_sentences')
train_sentence_pkl_path = sentence_pkl_base + '_train.pkl'
val_sentence_pkl_path = sentence_pkl_base + '_val.pkl'
test_sentence_pkl_path = sentence_pkl_base + '_test.pkl'
img_label_pkl_path = os.path.join(feat_dir, ds + '_img_labels.pkl')
img_simi_pkl_path = os.path.join(feat_dir, ds + '_img_simis.pkl')

image_h5_path = os.path.join(feat_dir, ds + '_images.h5')
image_h5_table = 'images'
name2id_map = os.path.join(feat_dir, ds + '_name2id.pkl')

# 结果评估相关的参数
result_root = opt.result_root
result_dir = prepare_dir(result_root, ds, trial_id)


with open(os.path.join(result_dir, 'args.txt'), 'w') as f:
    f.write(str(opt))

# checkpoint相关的参数
resnet_checkpoint = './models/resnet18-5c106cde.pth'  # 直接用pytorch训练的模型
vgg_checkpoint = './models/vgg16-397923af.pth'  # 从caffe转换而来

weight_pth_path = os.path.join(result_dir, ds + '_w.pth')
best_weight_pth_path = os.path.join(result_dir, ds + '_best_w.pth')
optimizer_pth_path = os.path.join(result_dir, ds + '_o.pth')
best_optimizer_pth_path = os.path.join(result_dir, ds + '_best_o.pth')


# 图示结果相关的超参数
visual_root = opt.visual_root
visual_dir = prepare_dir(visual_root, ds, trial_id)
