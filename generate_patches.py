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

parser = argparse.ArgumentParser(description='Generate ResDepth Training Patches')
parser.add_argument('dataset',
                type=str,
                help='Path to ODM dataset')
parser.add_argument('--photo-dem',
                type=str,
                default=default_photo_dem_path,
                help='Absolute path to photogrammetry generated DEM. Default: %(default)s')
parser.add_argument('--outdir',
                    type=str,
                    default=default_outdir,
                    help="Output directory where to store results. Default: %(default)s")

args = parser.parse_args()

dataset_path = args.dataset
photo_dem_path = os.path.join(dataset_path, default_photo_dem_path) if args.photo_dem == default_photo_dem_path else args.photo_dem

orthorectified_path = os.path.join(dataset_path, "orthorectified")
cwd_path = os.path.join(dataset_path, default_outdir) if args.outdir == default_outdir else args.outdir
if not os.path.exists(cwd_path):
    os.makedirs(cwd_path)

photo_dem_raster = rasterio.open(photo_dem_path)
print("Reading DEM: %s" % photo_dem_path)

crs = photo_dem_raster.profile.get('crs')
if crs is None:
    print("Whoops! DEM has no CRS, exiting...")
    print(1)

print("DEM CRS: %s" % str(crs))

photo_dem = photo_dem_raster.read()[0]
photo_dem_has_nodata = photo_dem_raster.profile.get('nodata') is not None

if photo_dem_has_nodata:
    photo_dem = ma.array(photo_dem, mask=photo_dem==photo_dem_raster.nodata)

photo_dem_stddev = np.std(photo_dem)
photo_dem_mean = np.mean(photo_dem)
print("DEM average: %.2f meters" % photo_dem_mean)

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
            
            # Quick check...
            if w1.width != w2.width or w2.height != w2.height or photo_dem_win.height != w1.height or photo_dem_win.width != w1.width:
                print("WARNING: window bounds do not match! Skipping...")
                print(w1, w2, photo_dem_win)
                continue

            # Write stacked image
            photo_dem_transform = photo_dem_raster.profile['transform']
            offset_x, offset_y = photo_dem_raster.xy(photo_dem_win.row_off, photo_dem_win.col_off, offset='ul')

            profile = {
                'driver': 'GTiff',
                'width': w1.width,
                'height': w1.height,
                'count': 3,
                'dtype': 'float32',
                'transform': rasterio.transform.Affine(photo_dem_transform[0], photo_dem_transform[1], offset_x, 
                                                        photo_dem_transform[3], photo_dem_transform[4], offset_y),
                'nodata': None,
                'crs': crs
            }

            outfile = os.path.join(cwd_path, "%s-%s" % (im1, im2))
            if not outfile.endswith(".tif"):
                outfile = outfile + ".tif"

            with rasterio.open(outfile, 'w', **profile) as wout:
                # Extract windows from images, DEM
                rgb1 = im1_rast.read(window=w1, indexes=(1, 2, 3, ))
                rgb2 = im2_rast.read(window=w2, indexes=(1, 2, 3, ))
                photo_dem_values = photo_dem_raster.read(1, window=photo_dem_win)

                wout.write(rgb2gray(rgb1), 1)
                wout.write(rgb2gray(rgb2), 2)
                wout.write(normalize_dem(photo_dem_values, photo_dem_stddev, photo_dem_mean), 3)
            
            exit(1)
photo_dem_raster.close()
