# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT
# (Modified) Koki Yoshida and Chenduo Huang
# 2017-06-01

"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import random
import scipy.misc
import numpy as np
import time
import os
import tensorflow as tf

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# Given a trained model, this function completes the images with specified mask
def complete_images(model, num_iters, input_image_paths, mask, output_dir,\
                    adam_config, save_per_num_iters=100, log_l1_loss=False): 
    image_shape = imread(input_image_paths[0]).shape
    # Assumes output images are square images
    image_size, num_imgs = image_shape[0], len(input_image_paths)

    start_time = time.time()
    if log_l1_loss:
        f = open('./log.txt', 'w')

    batch_idxs = int(np.ceil(num_imgs/model.batch_size)) 
    for idx in range(1):
        last_batch = idx == batch_idxs -1
        lower_bound = idx * model.batch_size
        upper_bound = num_imgs if last_batch else (idx+1) * model.batch_size
        cur_size = upper_bound - lower_bound
        
        cur_batch = input_image_paths[lower_bound : upper_bound]
        cur_images = [get_image(cur_path, image_size, is_crop=model.is_crop) \
                        for cur_path in cur_batch]
        cur_images = np.array(cur_images).astype(np.float32)
        if cur_size < model.batch_size:
            print("Padding the last batch with dummy images...")
            pad_size = ((0, int(model.batch_size - cur_size)), (0,0), (0,0), (0,0))
            cur_images = np.pad(cur_images, pad_size, 'constant').astype(np.float32)
      
        batch_mask = np.resize(mask, [model.batch_size] + list(image_shape))
        masked_images = np.multiply(cur_images, batch_mask)
        input_z = np.random.uniform(-1, 1, size=(model.batch_size, model.z_dim))
        # For Adam optimizer update on input noises
        m, v = 0, 0
        
        for i in range(num_iters):
            loss, g, G_imgs, contextual_loss = model.step_completion(input_z, batch_mask, cur_images)
            if log_l1_loss:
                f.write('%5.2f,%5.2f\n' % ((time.time() - start_time), np.mean(contextual_loss[:cur_size])))
            
            beta1, beta2 = adam_config['beta1'], adam_config['beta2']
            lr, eps = adam_config['lr'], adam_config['eps'] 
            m_prev, v_prev = np.copy(m), np.copy(v)
            m = beta1 * m_prev + (1 - beta1) * g[0]
            v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
            m_hat = m / (1 - beta1 ** (i + 1))
            v_hat = v / (1 - beta2 ** (i + 1))
            input_z += - np.true_divide(lr * m_hat, (np.sqrt(v_hat) + eps))
            input_z = np.clip(input_z, -1, 1)
            
            if i % save_per_num_iters == 0:
                cur_time = time.time()
                diff = cur_time - start_time
                print("After %d iterations(%5.2f), current average loss of batch %d is: %f" %\
                        (i, diff, idx, np.mean(loss[:cur_size])))
                batch_dir = os.path.join(output_dir, 'batch_idx_%d' % idx)
                zhats_dir = os.path.join(batch_dir, 'zhats_iter_%d' % i)
                completed_dir = os.path.join(batch_dir, 'completed_iter_%d' % i)
                os.makedirs(batch_dir, exist_ok=True)
                os.makedirs(zhats_dir, exist_ok=True)
                os.makedirs(completed_dir, exist_ok=True)
                
                completed_images = masked_images + np.multiply(G_imgs, 1.0 - batch_mask)
                for path_idx, path in enumerate(cur_batch):
                    zhats_image_out_path = os.path.join(zhats_dir, str(path_idx)+'.png')
                    completed_image_out_path = os.path.join(completed_dir, str(path_idx)+'.png')
                    save_image(G_imgs[path_idx, :, :, :], zhats_image_out_path)
                    save_image(completed_images[path_idx, :, :, :], completed_image_out_path)
    if log_l1_loss:
        f.close()

def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

def save_image(image, image_path):
    return scipy.misc.imsave(image_path, inverse_transform(image))

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.
