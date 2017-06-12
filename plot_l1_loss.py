#!/usr/bin/env python2
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

IMG_DIR = "./plots"

def read_lines(f, t, l):
    lines = f.readlines()[:-1]
    for i, line in enumerate(lines):
        cur_t, cur_l = line.split(',')
        t.append(i)
        l.append(float(cur_l))

def main():
    parser = argparse.ArgumentParser(description="Plot script for plotting L1 loss.")
    parser.add_argument('--save', dest='save_imgs', action='store_true',
                        help="Set this to true to save images under specified output directory.")
    parser.add_argument('--output', dest='output_dir', default=IMG_DIR,
                        help="Directory to store plots.")
    parser.add_argument('--print-stats', dest='print_stats', action='store_true',
                        help="Flag for priting L1 distance stats.")
    args = parser.parse_args()
    save_imgs = args.save_imgs
    output_dir = args.output_dir

    time, loss = [], []
    f = open('log.txt', 'r')
    read_lines(f, time, loss)
    
    plt.plot(time, loss, 'b--', label='Average L1 distance between outputs and originals')
    plt.legend(loc='upper left')
    plt.xlim([0, max(time)])
    plt.ylim([0, max(loss)])
    plt.xlabel('Num Iters')
    plt.ylabel('Average L1 loss')

    if save_imgs:
        # Save images to figure/
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + '/loss')
    else:
        plt.show()
    
    f.close()

    if args.print_stats:
        # Print loss summary (assuming 4000 iters ran)
        num_pxls, l = 64*64, loss
        print('Max | Min | Mean | Std. | Pxl Avg')
        print('------500------')
        print(max(l[:500]), min(l[:500]), np.mean(l[:500]), np.std(l[:500]), np.mean(l[:500])/num_pxls)
        print('------1000------')
        print(max(l[:1000]), min(l[:1000]), np.mean(l[:1000]), np.std(l[:1000]), np.mean(l[:1000])/num_pxls)
        print('------2000------')
        print(max(l[:2000]), min(l[:2000]), np.mean(l[:2000]), np.std(l[:2000]), np.mean(l[:2000])/num_pxls)
        print('------3000------')
        print(max(l[:3000]), min(l[:3000]), np.mean(l[:3000]), np.std(l[:3000]), np.mean(l[:3000])/num_pxls)
        print('------4000------')
        print(max(l[:4000]), min(l[:4000]), np.mean(l[:4000]), np.std(l[:4000]), np.mean(l[:4000])/num_pxls)
    


if __name__ == "__main__":
    main()
