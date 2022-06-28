import grp
from typing import Dict
from osgeo import gdal
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from src.data_utils import data_processing as dp
from src.data_utils.data_processing import make_model_dataset
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import random
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import warnings
import pickle
warnings.filterwarnings("ignore")

# def wrap_torch_datset(X, y, device):


def assemble_numpy_ds(blocks: dict, target: dict, stations_pixs: dict) -> tuple:
    """Assembles numpy dataset
        !!! CRUNCH ATTENTION !!!!
        !!! SINCE CURRENT WIND SPEED DATA IS NOT ALIGNED WITH THE OTHER ON TIME!!!!!
        stacks all available data into nd tensor, tiles elevation
        matches target and objects on pixels of stations

    Args:
        blocks (dict): block data from climate model
        target (dict): target from stations. see `get_y` function
        stations_pixs (dict): pixels of stations

    Returns:
        tuple: X, y
    """
    X = {}
    y = {}
    for k in tqdm(target.keys()):
        X_i = []
        for fn in blocks.keys():
            curr_pix = stations_pixs[k.casefold()]
            if curr_pix in blocks[fn].keys():
                X_i.append(blocks[fn][curr_pix])
        if len(X_i) > 0:
            y_i = target[k]
            X_i = sorted(X_i, key=lambda x: x.shape[0])
            wind_len = X_i[1].shape[0] #!!!!!!!! CRUNCH SINCE CURRENT WIND SPEED DATA IS NOT ALIGNED WITH THE OTHER ON TIME!!!!!
            X_i = [x[:wind_len, :, :] for x in X_i]
            for X_i_idx in range(len(X_i)):
                if X_i[X_i_idx].shape[0] == 1:
                    X_i[X_i_idx] = np.array([X_i[X_i_idx] for idx in range(X_i[-1].shape[0])]).squeeze() # repeating elevation
            X_i = np.stack(X_i, axis=1)
            # X.append(X_i)
            # y.append(y_i[:wind_len])
            X[k] = X_i
            y[k] = y_i[:wind_len]
    # X = np.concatenate(X)
    # y = np.concatenate(y)
    return (X, y)

def get_y(df: pd.DataFrame, start: str, end: str, speed_th: float = 20., station_name: str =None) -> dict:
    """Returns target variable for objects

    Args:
        df (pd.DataFrame): weather stations data
        start (str): start date. YYYY-MM-DD
        end (str): end date. YYYY-MM-DD
        speed_th (float, optional): threshold value for binary classification. Defaults to 20..
        station_name (str, optional): name of the station in russian, not sensitive to case. Defaults to None.

    Returns:
        dict: {station_name: indicators of exceeding threshold}
    """
    # speed_th = 10
    df['Дата'] = pd.to_datetime((df['Дата']), format="%Y/%m/%d")
    df_start_end = df.loc[(df['Дата'] >= pd.to_datetime(start)) & (df['Дата'] <= pd.to_datetime(end))]
    df_start_end = df_start_end[['Название метеостанции', 'Максимальная скорость', 'Средняя скорость ветра']]
    df_start_end['y'] = np.maximum(df_start_end['Максимальная скорость'],df_start_end['Средняя скорость ветра']) > speed_th
    df_start_end.drop(columns=['Максимальная скорость', 'Средняя скорость ветра'], inplace=True)
    if station_name is not None:
        grpb =  df_start_end.groupby(df_start_end['Название метеостанции'])
        assert station_name in grpb.groups.keys(), "No such station found. Available: " + "; ".join(list(grpb.groups.keys()))

        y = {station_name: grpb.get_group(station_name).y.values}
    else:
        grpb =  df_start_end.groupby(df_start_end['Название метеостанции'])
        ks = grpb.groups.keys()

        y = {k: df_start_end.groupby(df_start_end['Название метеостанции']).get_group(k).y.values for k in ks}


    return y

def get_pixel_stations(path_to_tifs: list, feature_names: list, station_names: list, station_list: pd.DataFrame) -> dict:
    """maps stations to the pixels in .tifs

    Args:
        path_to_tifs (list): path to tif files to get sample dataset - for shape
        feature_names (list): feature names - to get sample dataset
        station_names (list): list of station names
        station_list (pd.DataFrame): pandas table with information about stations

    Returns:
        dict: {station_name: (pixel coords)}
    """

    file_paths = [path_to_tifs[:-5]+ '/' + fn for fn in os.listdir(path_to_tifs[:-5]) if (fn[-4:] == '.tif' ) and feature_names[0] in fn]   
    dataset = gdal.Open(file_paths[0], gdal.GA_ReadOnly)
    stations_pixs = {}
    for station_name in station_names:
        
        pix = dp.closest_pixel_for_station(station_name=station_name, dataset=dataset, station_list=station_list)
        stations_pixs[station_name.casefold()] = pix
    return stations_pixs
def make_blocks(feature_names_list: list, path_to_tifs_list: str, half_side_size: int = 4, cmip: np.ndarray = None, verbose: bool = False, dset_num: int = 0) -> dict:
    """slices blocks from .tif data

    Args:
        feature_names (list): list of feature names in .tif files' names, [[`folder_i_features`] for i folder num]
        path_to_tifs (str): path to .tif files [path_to_tifs_i for i in folder num]
        half_side_size (int, optional): square block half size. Defaults to 4.
        verbose (bool, optional): if to print progress. Defaults to False.
        dset_num (int, optional): number of .tif file to pick from `path_to_tifs` if several present. Defaults to 0.

    Returns:
        dict: {`block center pixel`: surrounding 3d tensor}
    """
    print("Reading from .tifs")
    nps = {}
    for feature_names, path_to_tifs in zip(feature_names_list, path_to_tifs_list):
        nps = {**nps, **dp.get_nps(feature_names, path_to_tifs, verbose, dset_num=0)}
    print(".tifs has been read")
    
    slices_dict = {k: {} for k in nps.keys()} #key = center of block
    if cmip is not None:
        slices_dict['wind'] = {}
        for i in range(half_side_size, cmip.shape[1] - half_side_size):
            for j in range(half_side_size, cmip.shape[2] - half_side_size):
                slices_dict['wind'][(i, j)] = cmip[:, i - half_side_size : i + half_side_size, j - half_side_size : j + half_side_size]
    for k in tqdm(nps.keys()):
        np_ = nps[k]
        for i in range(half_side_size, np_.shape[1] - half_side_size):
            for j in range(half_side_size, np_.shape[2] - half_side_size):
                slices_dict[k][(i, j)] = np_[:, i - half_side_size : i + half_side_size, j - half_side_size : j + half_side_size]
    return slices_dict

