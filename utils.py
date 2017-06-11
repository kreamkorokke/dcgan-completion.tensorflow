# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT

"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def complete_images(model, num_iters, input_image_paths, mask, output_dir,\
                    adam_config, save_per_num_iters=100):
    os.makedirs(os.path.join(output_dir, 'hats_imgs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'completed'), exist_ok=True)

    tf.global_variables_initializer().run()
    image_shape = imread(input_image_paths[0])
    # Assumes output images are square images
    image_size, num_imgs = image_shape[0], len(input_image_paths)

    batch_idxs = int(np.ceil(num_imgs/model.batch_size)) 
    for idx in range(batch_idxs):
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

	batch_mask = np.resize(mask, [model.batch_size] + image_shape)
        masked_images = np.multiply(cur_images, batch_mask)
	input_z = np.random.uniform(-1, 1, size=(model.batch_size, model.z_dim))
        # For Adam optimizer update on input noises
	m, v = 0, 0

	for i in range(num_iters):
	    fd = {
		model.z: input_z,
		model.mask: batch_mask,
		model.images: cur_images,
		model.is_training: False
	    }
	    run = [model.complete_loss, model.grad_complete_loss, model.G]
	    loss, g, G_imgs = model.sess.run(run, feed_dict=fd)
            
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
                print("After %d iterations, current average loss of batch %d is: %f" %\
                        (i, idx, np.mean(loss[:cur_size]))
		batch_dir = os.path.join(output_dir,'batch_idx_%d' % idx)
                zhats_dir = os.path.join(batch_dir, 'zhats_iter_%d' % i)
                completed_dir = os.path.join(batch_dir, 'completed_iter_%d' % i)
                completed_images = masked_images + np.multiply(G_imgs, 1.0 - batch_mask)
                for path_idx, path in enumerate(cur_batch):
                    zhats_image_out_path = os.path.join(zhats_dir, str(path_idx)+'.png')
                    completed_image_out_path = os.path.join(completed_dir, str(path_idx)+'.png')
                    save_image(G_img[path_idx, :, :, :], zhats_image_out_path)
                    save_image(completed_images[path_idx, :, :, :], completed_image_out_path)


def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

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


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
