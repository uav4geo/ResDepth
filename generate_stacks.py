#!/usr/bin/env python3
# Author: Piero Toffanin
# License: AGPLv3

import os
import sys
sys.path.insert(0, os.path.join("..", "..", os.path.dirname(__file__)))

import rasterio
import numpy as np
import multiprocessing
import numpy.ma as ma
from osgeo import ogr
import argparse
from opensfm import dataset, tracking
from opensfm.reconstruction import compute_image_pairs

default_photo_dem_path = "odm_dem/dsm.tif"
default_outdir = "resdepth"
default_dt_dem_path = os.path.join(default_outdir, "ground_truth.tif")

parser = argparse.ArgumentParser(description='Generate ResDepth Training Stacked Images')
parser.add_argument('dataset',
                type=str,
                help='Path to ODM dataset')
parser.add_argument('--photo-dem',
                type=str,
                default=default_photo_dem_path,
                help='Absolute path to photogrammetry generated DEM. Default: %(default)s')
parser.add_argument('--gt-dem',
                type=str,
                default=default_dt_dem_path,
                help='Absolute path to ground truth DEM. Default: %(default)s')
parser.add_argument('--outdir',
                    type=str,
                    default=default_outdir,
                    help="Output directory where to store results. Default: %(default)s")

args = parser.parse_args()

dataset_path = args.dataset
photo_dem_path = os.path.join(dataset_path, default_photo_dem_path) if args.photo_dem == default_photo_dem_path else args.photo_dem
gt_dem_path = os.path.join(dataset_path, default_dt_dem_path) if args.gt_dem == default_dt_dem_path else args.gt_dem

orthorectified_path = os.path.join(dataset_path, "orthorectified")
cwd_path = os.path.join(dataset_path, default_outdir) if args.outdir == default_outdir else args.outdir
if not os.path.exists(cwd_path):
    os.makedirs(cwd_path)

stacks_path = os.path.join(cwd_path, "stacks")
if not os.path.exists(stacks_path):
    os.makedirs(stacks_path)

# Read photogrammetry DEM
if not os.path.exists(photo_dem_path):
    print("Whoops! Run OpenDroneMap with the --dsm option to generate a DEM, or provide a valid path to a DEM.")
    exit(1)

photo_dem_raster = rasterio.open(photo_dem_path)
print("Reading photogrammetry DEM: %s" % photo_dem_path)

crs = photo_dem_raster.profile.get('crs')
if crs is None:
    print("Whoops! DEM has no CRS, exiting...")
    print(1)

print("CRS: %s" % str(crs))

photo_dem = photo_dem_raster.read()[0]
photo_dem_has_nodata = photo_dem_raster.profile.get('nodata') is not None

if photo_dem_has_nodata:
    photo_dem = ma.array(photo_dem, mask=photo_dem==photo_dem_raster.nodata)

# Read ground truth DEM
if not os.path.exists(gt_dem_path):
    print("Whoops! %s does not exist. We need a valid ground truth DEM (use --gt-dem)." % gt_dem_path)
    exit(1)

gt_dem_raster = rasterio.open(gt_dem_path)
print("Reading ground truth DEM: %s" % gt_dem_path)
print("Dimensions: %sx%s" % (photo_dem.shape[1], photo_dem.shape[0]))

# Quick check of CRS...
gt_crs = gt_dem_raster.profile.get('crs')
if str(gt_crs) != str(crs):
    print("Uh oh! The CRS for the ground truth DEM does not match the CRS for the photogrammetry DEM (%s vs %s)" % (gt_crs, crs))
    exit(1)

gt_dem = gt_dem_raster.read()[0]
gt_dem_has_nodata = gt_dem_raster.profile.get('nodata') is not None

if gt_dem_has_nodata:
    gt_dem = ma.array(gt_dem, mask=gt_dem==gt_dem_raster.nodata)

# Compute globals
photo_dem_mean = np.mean(photo_dem)
gt_dem_mean = np.mean(gt_dem)
global_dem_mean = (photo_dem_mean + gt_dem_mean) / 2.0
global_dem_stddev = ma.concatenate((photo_dem, gt_dem, ), axis=None).std()

print("Average elevation: %.2f meters" % global_dem_mean)
print("Standard deviation: %.2f meters" % global_dem_stddev)
print("Dimensions: %sx%s" % (photo_dem.shape[1], photo_dem.shape[0]))

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
    lr = rast.xy(win.row_off + win.height, win.col_off + win.width)
    ll = rast.xy(win.row_off + win.height, win.col_off)
    return [ul, ur, lr, ll]

def poly_from_bounds(bounds):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for p in bounds:
        ring.AddPoint(p[0], p[1])
    ring.AddPoint(bounds[0][0], bounds[0][1])

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly

def intersect_bounds(a, b):
    p1 = poly_from_bounds(a)
    p2 = poly_from_bounds(b)
    i = p1.Intersection(p2)
    e = i.GetEnvelope()
    return [(e[0], e[3]), (e[1], e[3]), ([e[1], e[2]]), (e[0], e[2])]

def win_from_bounds(bounds, rast):
    # pixel window from geographic coordinate bounds
    b = [rast.index(*p) for p in bounds]
    return rasterio.windows.Window(
        b[0][1],
        b[0][0],
        b[1][1] - b[0][1],
        b[3][0] - b[0][0],
    )

def rgb2gray(rgb):
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.float32)

def normalize_dem(values, global_std, global_mean):
    values -= global_mean
    values *= global_std
    return values 

def bbox(img, nodata=0):
    a = np.where(img != nodata)
    bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
    return bbox

def crop(img, box):
    return img[box[0]:box[2]+1,box[1]:box[3]+1]

for im1, im2 in pairs:
    if im1 in remaining_images and im2 in remaining_images:
        remaining_images.remove(im1)
        remaining_images.remove(im2)

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

            # Quick checks..
            if im1_rast.profile['count'] != 4:
                print("Expecting RGBA images but %s is not." % im1)
            if im2_rast.profile['count'] != 4:
                print("Expecting RGBA images but %s is not." % im2)
            
            im1_alpha = im1_rast.read(4)
            im2_alpha = im2_rast.read(4)

            im1_win = rasterio.windows.get_data_window(im1_alpha, nodata=0)
            im2_win = rasterio.windows.get_data_window(im2_alpha, nodata=0)

            # Compute intersection of the two photos
            bounds = intersect_bounds(compute_bounds(im1_rast, im1_win), compute_bounds(im2_rast, im2_win))
            
            # Compute windows for images and DEM
            w1 = win_from_bounds(bounds, im1_rast)
            w2 = win_from_bounds(bounds, im2_rast)
            photo_dem_win = win_from_bounds(bounds, photo_dem_raster)
            gt_dem_win = win_from_bounds(bounds, gt_dem_raster)
            
            # Quick check...
            if w1.width != w2.width or w2.height != w2.height or photo_dem_win.height != w1.height or photo_dem_win.width != w1.width or gt_dem_win.width != w1.width or gt_dem_win.height != w2.height:
                print("WARNING: window bounds do not match! Skipping...")
                print(w1, w2, photo_dem_win, gt_dem_win)
                continue
            
            # Extract windows from images, DEMs
            rgb1 = im1_rast.read(window=w1, indexes=(1, 2, 3, ))
            rgb2 = im2_rast.read(window=w2, indexes=(1, 2, 3, ))
            photo_dem_values = photo_dem_raster.read(1, window=photo_dem_win)
            gt_dem_values = gt_dem_raster.read(1, window=gt_dem_win)

            gray1 = rgb2gray(rgb1)
            gray2 = rgb2gray(rgb2)

            # Crop
            blank_area = np.logical_or(gray1==0,
                         np.logical_or(gray2==0,
                         np.logical_or(photo_dem_values==photo_dem_raster.nodata, gt_dem_values==gt_dem_raster.nodata)))

            gray1[blank_area] = 0
            gray2[blank_area] = 0
            photo_dem_values[blank_area] = photo_dem_raster.nodata
            gt_dem_values[blank_area] = gt_dem_raster.nodata
            box = bbox(gray1)

            gray1 = crop(gray1, box)
            gray2 = crop(gray2, box)
            photo_dem_values = crop(photo_dem_values, box)
            gt_dem_values = crop(gt_dem_values, box)

            # Don't normalize nodata values
            nodata_area = photo_dem_values==photo_dem_raster.nodata
            normalized_photo_dem_values = normalize_dem(photo_dem_values, global_dem_stddev, global_dem_mean)
            normalized_photo_dem_values[nodata_area] = photo_dem_raster.nodata

            nodata_area = gt_dem_values==gt_dem_raster.nodata
            normalized_gt_dem_values = normalize_dem(gt_dem_values, global_dem_stddev, global_dem_mean)
            normalized_gt_dem_values[nodata_area] = gt_dem_raster.nodata

            # Write stacked image
            photo_dem_transform = photo_dem_raster.profile['transform']
            offset_x, offset_y = photo_dem_raster.xy(photo_dem_win.row_off + box[0], photo_dem_win.col_off + box[1], offset='ul')

            profile = {
                'driver': 'GTiff',
                'width': gray1.shape[1],
                'height': gray1.shape[0],
                'count': 4,
                'dtype': 'float32',
                'transform': rasterio.transform.Affine(photo_dem_transform[0], photo_dem_transform[1], offset_x, 
                                                        photo_dem_transform[3], photo_dem_transform[4], offset_y),
                'nodata': None,
                'crs': crs
            }

            outfile = os.path.join(stacks_path, "%s-%s" % (im1, im2))
            if not outfile.endswith(".tif"):
                outfile = outfile + ".tif"

            with rasterio.open(outfile, 'w', **profile) as wout:
                wout.write(gray1, 1)
                wout.write(gray2, 2)
                wout.write(normalized_photo_dem_values, 3)
                wout.write(normalized_gt_dem_values, 4)
            
            print("Wrote %s" % outfile)
photo_dem_raster.close()
