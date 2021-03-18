#!/usr/bin/env python3
# Author: Piero Toffanin
# License: AGPLv3

import os
import sys
sys.path.insert(0, os.path.join("..", "..", os.path.dirname(__file__)))

import rasterio
import numpy as np
import multiprocessing
import argparse
from opensfm import dataset, tracking
from opensfm.reconstruction import compute_image_pairs

default_photo_dem_path = "odm_dem/dsm.tif"

parser = argparse.ArgumentParser(description='Generate ResDepth Training Patches')
parser.add_argument('dataset',
                type=str,
                help='Path to ODM dataset')
parser.add_argument('--photo-dem',
                type=str,
                default=default_photo_dem_path,
                help='Absolute path to photogrammetry generated DEM. Default: %(default)s')

args = parser.parse_args()

dataset_path = args.dataset
photo_dem_path = os.path.join(dataset_path, default_photo_dem_path) if args.photo_dem == default_photo_dem_path else args.photo_dem

orthorectified_path = os.path.join(dataset_path, "orthorectified")

photo_dem_raster = rasterio.open(photo_dem_path)
print("Reading DEM: %s" % photo_dem_path)
photo_dem = photo_dem_raster.read()[0]
photo_dem_h, photo_dem_w = photo_dem.shape
print("DEM dimensions: %sx%s" % (photo_dem_w, photo_dem_h))

# Read reconstruction
data = dataset.DataSet(os.path.join(dataset_path, "opensfm"))
udata = dataset.UndistortedDataSet(data)
tracks_manager = udata.load_undistorted_tracks_manager()
reconstructions = udata.load_undistorted_reconstruction()
if len(reconstructions) == 0:
    raise Exception("No reconstructions available")

reconstruction = reconstructions[0]
images = tracks_manager.get_shot_ids()
remaining_images = set(images)
common_tracks = tracking.all_common_tracks(tracks_manager)
camera_priors = data.load_camera_models()
print("Computing image pairs...")
pairs = compute_image_pairs(common_tracks, camera_priors, data)


def compute_bounds(rast, win):
    ul = rast.xy(win.row_off, win.col_off)
    ur = rast.xy(win.row_off, win.col_off + win.width)
    ll = rast.xy(win.row_off + win.height, win.col_off)
    lr = rast.xy(win.row_off + win.height, win.col_off + win.width)
    return [ul, ur, ll, lr]

def intersect_bounds(a, b):
    # TODO!
    ul = (max(a[0], b[0])

for im1, im2 in pairs:
    if im1 in remaining_images and im2 in remaining_images:
        remaining_images.remove(im1)
        remaining_images.remove(im2)
        print("(%s, %s)" % (im1, im2))

        # Check if both images have been orthorectified
        im1_rect = os.path.join(orthorectified_path, im1 + ".tif")
        im2_rect = os.path.join(orthorectified_path, im2 + ".tif")

        for p in [im1_rect, im2_rect]:
            if not os.path.exists(p):
                # TODO: automatically orthorectify?
                print("%s does not exist (we need that!). Run ODM's orthorectification tool first. Skipping...")
                continue

            # Read images
            im1_rast = rasterio.open(im1_rect)
            im2_rast = rasterio.open(im2_rect)

            im1_data = im1_rast.read()
            im2_data = im2_rast.read()

            # Quick checks..
            if im1_data.shape[0] != 4:
                print("Expecting RGBA images but %s is not." % im1)
            if im2_data.shape[0] != 4:
                print("Expecting RGBA images but %s is not." % im2)
            
            im1_alpha = im2_data[-1]
            im2_alpha = im2_data[-1]
            im1_win = rasterio.windows.get_data_window(im1_alpha, nodata=0)
            im2_win = rasterio.windows.get_data_window(im2_alpha, nodata=0)
            # Compute intersection of the two photos

            
            print(intersect_bounds(compute_bounds(im1_rast, im1_win), compute_bounds(im2_rast, im2_win)))
            exit(1)
# for shot in reconstruction.shots.values():
#     print(shot.id)

photo_dem.close()
