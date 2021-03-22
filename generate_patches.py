#!/usr/bin/env python3
# Author: Piero Toffanin
# License: AGPLv3

import os
import glob
import sys
sys.path.insert(0, os.path.join("..", "..", os.path.dirname(__file__)))

import rasterio
import numpy as np
import argparse
import warnings
import random
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

default_outdir = os.path.join("..", "patches")
parser = argparse.ArgumentParser(description='Generate ResDepth Training Patches From Stacks')
parser.add_argument('stacks',
                type=str,
                help='Path to stack images directory')
parser.add_argument('--patch-size',
                type=int,
                default=128,
                help='Patch size. Default: %(default)s')
parser.add_argument('--outdir',
                    type=str,
                    default=default_outdir,
                    help="Output directory where to store results. Default: %(default)s")

args = parser.parse_args()

stacks_path = args.stacks
cwd_path = os.path.join(stacks_path, default_outdir) if args.outdir == default_outdir else args.outdir
if not os.path.exists(cwd_path):
    os.makedirs(cwd_path)

images = glob.glob(os.path.join(stacks_path, "*.tif"))
if len(images) == 0:
    print("No .tif images found in %s. Did you run generate_stacks.py first?" % stacks_path)

print("Found %s .tif images" % len(images))
for im_path in images:
    im_filename = os.path.basename(im_path)
    im_base, _ = os.path.splitext(im_filename)

    def write_patch(win, name):
        profile = {
            'driver': 'GTiff',
            'width': win.shape[2],
            'height': win.shape[1],
            'count': win.shape[0],
            'dtype': win.dtype.name,
            'transform': None,
            'nodata': None
        }
        outfile = os.path.join(cwd_path, "%s-%s.tif" % (im_base, name))

        with rasterio.open(outfile, 'w', **profile) as fout:
            for b in range(win.shape[0]):
                fout.write(win[b], b + 1)

    print("Processing %s" % im_filename)
    min_cells = (args.patch_size * args.patch_size) / 4
    with rasterio.open(im_path) as f:
        data = f.read()

        def write_window(i, j, rotate=None, vflip=False, hflip=False):
            if f.width - i < args.patch_size:
                i = f.width - args.patch_size
            if f.height - j < args.patch_size:
                j = f.height - args.patch_size
            win = data[:,j:j+args.patch_size,i:i+args.patch_size]

            # Skip empty patches or those without much data
            if np.count_nonzero(win[0]==0) > min_cells:
                return

            patch_name = "%s-%s" % (i, j)

            if rotate is not None:
                # Rotate
                patch_name += "-r%s" % rotate
                win = np.rot90(win, int(rotate / 90), axes=(1, 2))

            if vflip:
                patch_name += "-vflip"
                win = np.flip(win, axis=1)

            if hflip:
                patch_name += "-hflip"
                win = np.flip(win, axis=2)

            write_patch(win, patch_name)

        for i in range(0, f.width, args.patch_size):
            for j in range(0, f.height, args.patch_size):
                # Write sliding windows
                write_window(i, j)
                
                # Randomly rotate sliding window
                write_window(i, j, rotate=random.choice((90, 180, 270)))

                # Write flipped
                r = random.randint(0, 2)
                if r == 0:
                    write_window(i, j, vflip=True)
                elif r == 1:
                    write_window(i, j, hflip=True)

                # Write window with a random offset in X/Y
                rand_i = i + random.randint(0, args.patch_size - 1)
                rand_j = j + random.randint(0, args.patch_size - 1)
                write_window(rand_i, rand_j)

                # Randomly Rotate
                write_window(rand_i, rand_j, rotate=random.choice((90, 180, 270)))

                # Write flipped
                r = random.randint(0, 2)
                if r == 0:
                    write_window(rand_i, rand_j, vflip=True)
                elif r == 1:
                    write_window(rand_i, rand_j, hflip=True)
