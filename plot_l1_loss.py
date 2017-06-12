#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import argparse

IMG_DIR = "./plots"

def read_lines(f, t, l):
    lines = f.readlines()[:-1]
    for line in lines:
        cur_t, cur_l = line.split(',')
        t.append(float(cur_t))
        l.append(float(cur_l))

def main():
    parser = argparse.ArgumentParser(description="Plot script for plotting L1 loss.")
    parser.add_argument('--save', dest='save_imgs', action='store_true',
                        help="Set this to true to save images under specified output directory.")
    parser.add_argument('--output', dest='output_dir', default=IMG_DIR,
                        help="Directory to store plots.")
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
    plt.xlabel('Time (s)')
    plt.ylabel('Average L1 loss')

    if save_imgs:
        # Save images to figure/
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + "/" + attack_name)
    else:
        plt.show()
    
    f.close()


if __name__ == "__main__":
    main()
