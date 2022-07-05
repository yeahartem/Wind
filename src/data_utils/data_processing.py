from tkinter import Y
from osgeo import gdal

import matplotlib.pyplot as plt
import os, glob
import numpy as np
from tqdm import tqdm
import subprocess
import pandas as pd
from geopy.distance import great_circle

# from geopy.distance import geodesic
from math import sin, cos, sqrt, atan2, radians
from sklearn import preprocessing
from scipy import interpolate


def get_file_paths(
    path_to_data: str = "drive/MyDrive/Belgorodskaya/*.tif",
    feature_names: list = ["tmax", "tmin", "pr"],
):
    """
    Filters out required features amongs terraclim dataset
    Arguments:
      path_to_data (str): path to directory that containts terraclim dataset
      feature_names (list): list of required features

    Returns:
      dict: key -- feature name; value -- list of related tif files
    """
    files_to_mosaic = glob.glob(path_to_data)
    files_to_mosaic = list(
        filter(lambda x: sum(fn in x for fn in feature_names) > 0, files_to_mosaic)
    )
    file_paths = {
        fn: list(filter(lambda x: fn in x, files_to_mosaic)) for fn in feature_names
    }
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
    x_0, y_0, x_res, y_res = (
        coords_dict["x"],
        coords_dict["y"],
        coords_dict["x_res"],
        coords_dict["y_res"],
    )
    # coord = [37, 46.5]
    x_coords = np.array(range(raster_xsize)) * x_res + x_0
    y_coords = np.array(range(raster_ysize)) * y_res + y_0
    # for y in y_coords:
    #   for x in x_coords:
    #     print((x, y))
    # d = []
    R = 6371
    # R = 6373
    # coord0 = radians(coord[0])
    # coord1 = radians(coord[1])
    closest = 21212121
    I = 0
    J = 0
    for i, theta in enumerate(x_coords):
        for j, fi in enumerate(y_coords):
            # r = R*np.sqrt((theta - coord[0])**2 + np.cos((theta + coord[0])/2)**2*(fi - coord[1])**2)
            # r = geodesic((theta,fi), coord).kilometers
            r = great_circle((theta, fi), coord).kilometers
            # print(r)
            # theta = radians(theta)
            # fi = radians(fi)
            # dlon = fi - coord1
            # dlat = theta - coord0
            # a = sin(dlat / 2)**2 + cos(coord0) * cos(theta) * sin(dlon / 2)**2
            # r = R * 2 * atan2(sqrt(a), sqrt(1 - a))
            # d.append(r)
            if r < closest:
                closest = r
                I = i + 1
                J = j + 1
    # N = np.argmin(d)
    # closest_x_idx = (N//len(y_coords))
    # closest_y_idx = (- closest_x_idx)*len(y_coords) + N - 1
    closest_x_idx = I
    closest_y_idx = J
    return closest_x_idx, closest_y_idx


def closest_pixel_for_station(
    station_name: str, dataset: gdal.Dataset, station_list: pd.DataFrame
):
    """Finds the closest pixel indices in the dataset for the station
    Args:
      station_name (str): name of the station
      dataset (gdal.Dataset): dataset with pixels
      station_list (pd.DataFrame): table with stations' coordinates
    Returns:
      tuple: x, y indices among the dataset
    """
    station = station_list[station_list["Наименование станции"] == station_name]
    coord = [station["Долгота"].values[0], station["Широта"].values[0]]
    pix = get_closest_pixel(dataset=dataset, coord=coord)
    return pix


def make_model_dataset(
    station_name: str,
    start_date: str,
    end_date: str,
    wind_cmip: np.array,
    station_list: pd.DataFrame,
    path_to_history: str = "data/history",
    path_to_elev: str = "data/elev",
    features: list = ["tasmax", "tasmin", "pr"],
):

    table = pd.DataFrame(
        {"Date": pd.date_range(start="01.01.2006", end="01.31.2020", freq="D")}
    )
    for feature_name in features:
        file_paths = [
            path_to_history + "/" + fn
            for fn in os.listdir(path_to_history)
            if (fn[-4:] == ".tif") and feature_name in fn
        ]
        dataset = gdal.Open(file_paths[0], gdal.GA_ReadOnly)
        pix = closest_pixel_for_station(
            station_name=station_name, dataset=dataset, station_list=station_list
        )
        array = []
        for i in range(1, 5145):
            band = dataset.GetRasterBand(i)
            if feature_name in ["tasmax", "tasmin"]:
                arr = band.ReadAsArray()
                array.append(arr[pix[1]][pix[0]] - 273.15)
            else:
                arr = band.ReadAsArray()
                array.append(arr[pix[1]][pix[0]])
        table = table.join(pd.DataFrame({feature_name: array}))

    array_wind = []
    for day in wind_cmip:
        array_wind.append(day[pix[1]][pix[0]])
    table = table.join(pd.DataFrame({"CMIP_wind": array_wind}))

    file_paths = [path_to_elev + "/" + "elevation.tif"]
    dataset = gdal.Open(file_paths[0], gdal.GA_ReadOnly)
    array = []
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray()
    array.append(arr[pix[1]][pix[0]])
    table = table.join(pd.DataFrame({"el": 5144 * array}))
    table = table.set_index("Date")
    table = table.loc[table.index >= pd.to_datetime(start_date)]
    table = table.loc[table.index <= pd.to_datetime(end_date)]
    return table


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
        Lat[i] = coords_dict["y"] + i * coords_dict["y_res"]
    Lon = np.zeros((dset_tmp.RasterXSize))
    for i in range(Lon.shape[0]):
        Lon[i] = coords_dict["x"] + i * coords_dict["x_res"]
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
            tlxs.append(coords_dict["x"])
            tlys.append(coords_dict["y"])

    fig, ax = plt.subplots()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax.scatter(tlxs, tlys)

    for i in range(len(tlxs)):
        ax.annotate(i, (tlxs[i], tlys[i]))
    plt.gca().set_aspect("equal", adjustable="box")
    ax.set_title("Positions of top left corners of each raster")
    ax.grid(True)


def dataset_to_np(
    dataset: gdal.Dataset, x_off: int, y_off: int, xsize: int, ysize: int
):
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
        np_tmp = dataset_to_np(
            dsets[fn],
            x_off=0,
            y_off=0,
            xsize=dsets[fn].RasterXSize,
            ysize=dsets[fn].RasterYSize,
        )
        # Scaling in accordance with dataset description
        if fn == "tmin" or fn == "tmax":
            nps[fn] = np_tmp * 0.1
        elif fn == "ws":
            nps[fn] = np_tmp * 0.01
        elif fn == "vap":
            nps[fn] = np_tmp * 0.001
        elif fn == "seasurfacetemp":
            nps[fn] = np_tmp * 0.01
        else:
            nps[fn] = np_tmp

    # getting mean temp if accessible
    if "tmin" in feature_names and "tmax" in feature_names:
        nps["tmean"] = (nps["tmax"] + nps["tmin"]) / 2

    return nps


def cmiper(cmip, lat_left, lat_right, lon_left, lon_right):
    """
    Preprocess CMIP to understandable DataFrame.

    cmip - downloaded CMIP as a DataFrame.
    """
    a = cmip.loc[
        (cmip.lat_bnds >= lat_left - 1.5) & (cmip.lat_bnds <= lat_right + 1.5)
    ]  # Широта
    a = a.loc[(a.lon_bnds >= lon_left) & (a.lon_bnds <= lon_right)]  # Долгота

    a = a.reset_index()
    a.drop(
        columns=["time_bnds", "lat_bnds", "lon_bnds", "height", "bnds", "time"],
        inplace=True,
    )

    lat_idx = list(set(a.lat))
    lon_idx = list(set(a.lon))
    le_lat, le_lon = preprocessing.LabelEncoder(), preprocessing.LabelEncoder()
    le_lat.fit(lat_idx)
    le_lon.fit(lon_idx)

    a.insert(loc=0, column="lat_idx", value=le_lat.transform(a.lat))
    a.insert(loc=1, column="lon_idx", value=le_lon.transform(a.lon))
    a.drop(columns=["lat", "lon"], inplace=True)

    return a


def to_3dar(cmip_pr):
    """
    Preprocess CMIP to 3d np.array.

    cmip_pr - preprocessed cmip.
    """
    day = int(cmip_pr.shape[0] / 3650)
    lonn = cmip_pr.lon_idx.unique().shape[0]
    latt = cmip_pr.lat_idx.unique().shape[0]
    fin = []

    for i in range(3650):

        aa = cmip_pr[i * day : i * day + day]  # cannot put variable day here
        z = np.zeros((latt, lonn))
        for i, j, w in zip(aa["lat_idx"], aa["lon_idx"], aa["sfcWind"]):
            z[i, j] = w
        fin.append(z[1:-1])
    fin = np.stack(fin)

    x = np.arange(18, 170.01, 2)  # longitude   25
    y = np.arange(41.25, 78.74, 1.5)  # latitude

    xnew = np.arange(18, 170.01, 0.25)  # 25
    ynew = np.arange(41.25, 78.74, 0.25)  # [::-1]              # 19

    ss_x = np.where(((xnew >= 19) & (xnew <= 169)))[0]  # 22
    ss_y = np.where(((ynew >= 41) & (ynew <= 77)))[0]  # 15

    final = []
    for z in fin:
        f = interpolate.interp2d(x, y, z)  # or (x, y, z)
        final.append(f(xnew, ynew)[np.ix_(ss_y, ss_x)])
    final = np.stack(final)

    new_final = []
    for i in final:
        new_final.append(i[::-1])
    final = np.stack(new_final)

    # feb_28_in_2008_and_2012 = [788, 2248]

    feb_2012 = (final[2248] + final[2249]) / 2
    feb_2012 = np.expand_dims(feb_2012, axis=0)
    final = np.vstack((final[:2249], feb_2012, final[2249:]))

    feb_2008 = (final[788] + final[789]) / 2
    feb_2008 = np.expand_dims(feb_2008, axis=0)
    final = np.vstack((final[:789], feb_2008, final[789:]))

    return final


def all(cmip, lat_left, lat_right, lon_left, lon_right):
    """
    Consist of all needed functions for the preprocessing of CMIP.
    """

    intermediate = cmiper(cmip, lat_left, lat_right, lon_left, lon_right)
    return to_3dar(intermediate)
