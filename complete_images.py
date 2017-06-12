#!/usr/bin/env python3
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05
#
# (Modified) Koki Yoshida and Chenduo Huang
# License: MIT
# 2017-06-01


import argparse
import os
import tensorflow as tf

from dcgan import DCGAN
from img_utils import imread, complete_images
import numpy as np

def path_check_exist(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("Checkpoint directory %s does not exist!" % path)
    return path

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--num-iters', dest='num_iters', type=int, default=1000)
parser.add_argument('--img-size', dest='img_size', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--checkpoint-dir', dest='checkpoint_dir',\
                    type=path_check_exist, default='checkpoint')
parser.add_argument('--output-dir', dest='output_dir', type=str, default='completions')
parser.add_argument('--out-interval', dest='out_interval', type=int, default=100)
parser.add_argument('--mask-type', dest='mask_type', type=str,\
                    choices=['random', 'center', 'left', 'full', 'grid'],\
                    default='center')
parser.add_argument('--center-scale', dest='center_scale', type=float, default=0.25)
parser.add_argument('imgs', type=str, nargs='+')
parser.add_argument('--log-l1-loss', dest='log_l1_loss', action='store_true')

args = parser.parse_args()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    model = DCGAN(sess, image_size=args.img_size,
                  checkpoint_dir=args.checkpoint_dir, lam=args.lam)
    # Need to have a loaded model
    tf.global_variables_initializer().run()
    assert(model.load(model.checkpoint_dir))
    
    # Construct mask
    image_shape = imread(args.imgs[0]).shape
    mask_type = args.mask_type
    if mask_type == 'random':
        fraction_masked = 0.2
        mask = np.ones(image_shape)
        mask[np.random.random(image_shape[:2]) < fraction_masked] = 0.0
    elif mask_type == 'center':
        center_scale = args.center_scale
        assert(center_scale <= 0.5)
        mask = np.ones(image_shape)
        l = int(image_shape[0] * center_scale)
        u = int(image_shape[0] * (1.0 - center_scale))
        mask[l:u, l:u, :] = 0.0
    elif mask_type == 'left':
        mask = np.ones(image_shape)
        c = image_size // 2
        mask[:,:c,:] = 0.0
    elif mask_type == 'full':
        mask = np.ones(image_shape)
    elif mask_type == 'grid':
        mask = np.zeros(image_shape)
        mask[::4,::4,:] = 1.0
    else:
        raise "Unknown mask type %s" % mask_type

    adam_config = {'beta1': args.beta1, 'beta2': args.beta2, 'lr': args.lr, 'eps': args.eps}
    complete_images(model, num_iters=1000, input_image_paths=args.imgs, mask=mask,\
                    output_dir=args.output_dir, adam_config=adam_config,\
                    save_per_num_iters=args.out_interval, log_l1_loss=args.log_l1_loss)
