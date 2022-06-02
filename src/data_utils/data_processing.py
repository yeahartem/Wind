from osgeo import gdal

import matplotlib.pyplot as plt
import os, glob
import numpy as np
from tqdm import tqdm
import subprocess

def get_file_paths(path_to_data: str = 'drive/MyDrive/Belgorodskaya/*.tif', feature_names: list = ['tmax', 'tmin', 'pr']):
  """
  Filters out required features amongs terraclim dataset

  Arguments:
    path_to_data (str): path to directory that containts terraclim dataset
    feature_names (list): list of required features
    
  Returns:
    dict: key -- feature name; value -- list of related tif files
  """
  files_to_mosaic = glob.glob(path_to_data)
  files_to_mosaic = list(filter(lambda x: sum(fn in x for fn in feature_names) > 0, files_to_mosaic))
  file_paths = {fn: list(filter(lambda x: fn in x, files_to_mosaic)) for fn in feature_names}
  return file_paths

def get_closest_pixel(dataset: gdal.Dataset, coord: np.ndarray):
  """Finds the closest pixel indices in the dataset

  Args:
      dataset (gdal.Dataset): dataset with pixels
      coord (np.ndarray): coordinate for which the closest pixel's indices in the dataset will be found

  Returns:
      tuple: x, y indices among the dataset
  """
  coords_dict = get_coords_res(dataset)
  raster_xsize = dataset.RasterXSize
  raster_ysize = dataset.RasterYSize
  x_0, y_0, x_res, y_res = coords_dict['x'], coords_dict['y'], coords_dict['x_res'], coords_dict['y_res']
  # coord = [37, 46.5]
  x_coords = np.array(range(raster_xsize)) * x_res + x_0
  y_coords = np.array(range(raster_ysize)) * y_res + y_0
  closest_x_idx = np.argmin(np.abs(x_coords - coord[0]))
  closest_y_idx = np.argmin(np.abs(y_coords - coord[1]))
  return closest_x_idx, closest_y_idx



def get_coords_res(dataset: gdal.Dataset):
  """
  For given dataset returns position of top left corner and resolutions

  Arguments:
    dataset (osgeo.gdal.Dataset): gdal dataset

  Returns:
    dict: containts coordinates of top left corner and
       resolutions alog x and y axes
  """
  gt = dataset.GetGeoTransform()
  output = {}
  output["x"] = gt[0]
  output["y"] = gt[3]
  output["x_res"] = gt[1]
  output["y_res"] = gt[-1]
  return output

def extract_latitude_longtitute(path_to_tifs: str, feature_name: str):
  """
  Extract 1d arrays of longitutde and latitude of given .tif data
  Helps to build a mapping between raster spatial indices and coordinates on the earth

  Arguments:
    path_to_data (str): path to directory that containts terraclim dataset
    feature_names (str): feature name of the interest
    
  Returns:
    tuple: longtitude and latitude 1d arrays 
  """
  file_paths = get_file_paths(path_to_tifs, [feature_name])
  dset_tmp = gdal.Open(file_paths[feature_name][0])
  coords_dict = get_coords_res(dset_tmp)
  Lat = np.zeros((dset_tmp.RasterYSize))
  for i in range(Lat.shape[0]):
    Lat[i] = coords_dict['y'] + i * coords_dict['y_res']
  Lon = np.zeros((dset_tmp.RasterXSize))
  for i in range(Lon.shape[0]):
    Lon[i] = coords_dict['x'] + i * coords_dict['x_res']
  return Lat, Lon


def plot_tl_positions(file_paths: list):
  """
  Viualize positions of top left corners of dataset given 

  Arguments:
  file_paths (list): list of paths to files that contain datasets
  """
  tlxs = []
  tlys = []
  for fp in tqdm(file_paths):
    dataset = gdal.Open(fp, gdal.GA_ReadOnly) 
    if dataset is not None:
      coords_dict = get_coords_res(dataset)
      tlxs.append(coords_dict['x'])
      tlys.append(coords_dict['y'])
                          
  fig, ax = plt.subplots()
  fig.set_figheight(15)
  fig.set_figwidth(15)
  ax.scatter(tlxs, tlys)

  for i in range(len(tlxs)):
    ax.annotate(i, (tlxs[i], tlys[i]))
  plt.gca().set_aspect('equal', adjustable='box')
  ax.set_title("Positions of top left corners of each raster")
  ax.grid(True)


def dataset_to_np(dataset: gdal.Dataset, x_off: int, y_off: int, xsize: int, ysize: int):
  """
  Converts gdal.Dataset to numpy array
  !NB: raster bands are enumerated starting from 1!

  Arguments:
    dataset (gdal.Dataset): dataset to cast
    x_off (int): starting x position - idx
    y_off (int): starting y position - idx
    xsize (int): number of points to save in x direction
    ysize (int): number of points to save in y direction
  Returns:
    np.ndarray -- 3d tensor of information given in dataset
  """
  
  shape = [dataset.RasterCount, ysize, xsize]
  output = np.empty(shape)
  for r_idx in range(shape[0]):
    band = dataset.GetRasterBand(r_idx + 1)
    arr = band.ReadAsArray(x_off, y_off, xsize, ysize)
    output[r_idx, :, :] = np.array(arr)
  
  return output

def get_nps(feature_names, path_to_tifs, dset_num=0):
  file_paths = get_file_paths(path_to_tifs, feature_names)
  # open gdal files
  dsets = {}
  for fn in feature_names:
    dset = gdal.Open(file_paths[fn][dset_num])
    dsets[fn] = dset
  # reading into np, scaling in accordance with terraclim provided
  nps = {}
  for fn in feature_names:
    np_tmp = dataset_to_np(dsets[fn], x_off = 0, y_off = 0, xsize = dsets[fn].RasterXSize, ysize = dsets[fn].RasterYSize)
    #Scaling in accordance with dataset description
    if fn == 'tmin' or fn == 'tmax':
      nps[fn] = np_tmp * 0.1
    elif fn == 'ws':
      nps[fn] = np_tmp * 0.01
    elif fn == 'vap':
      nps[fn] = np_tmp * 0.001
    elif fn == 'seasurfacetemp':
      nps[fn] = np_tmp * 0.01
    else:
      nps[fn] = np_tmp
  
  #getting mean temp if accessible
  if 'tmin' in feature_names and 'tmax' in feature_names:
    nps['tmean'] = (nps['tmax'] + nps['tmin']) / 2

  return nps