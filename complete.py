#!/usr/bin/env python3
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05

import argparse
import os
import tensorflow as tf

from model import DCGAN
from utils import imread, complete_images

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--nIter', type=int, default=1000)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--output-dir', type=str, dest='output_dir', default='completions')
parser.add_argument('--outInterval', type=int, default=50)
parser.add_argument('--mask-type', type=str, dest='mask_type',
                    choices=['random', 'center', 'left', 'full', 'grid'],
                    default='center')
parser.add_argument('--center-scale', type=float, dest='center_scale', default=0.25)
parser.add_argument('imgs', type=str, nargs='+')

args = parser.parse_args()

assert(os.path.exists(args.checkpointDir))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=args.imgSize,
                  checkpoint_dir=args.checkpointDir, lam=args.lam)
    # Need to have a loaded model
    assert(dcgan.load())
    
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
    complete_images(dcgan, num_iters=1000, input_image_paths=args.imgs, mask=mask,\
                    output_dir=args.output_dir, )
