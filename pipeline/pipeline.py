import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
#matplotlib.use('Agg')
import random
from pytorch_lightning.loggers import TensorBoardLogger
import json
from tqdm import tqdm
import imageio
import shap
import sys
sys.path.append('..')
import os
import warnings
import pickle
warnings.filterwarnings("ignore")
from collections import OrderedDict

import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as T

# from src.data_assemble.assemble_ml import *
# from src.data_assemble.assemble_conv import *
# from src.models.utils import *
from src.data_assemble.wrap_data import *
from src.models.WindCNN import *
from src.data_assemble.wrap_data import *

# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
random.seed(24)


def generate_dataset(
    feature_names: list = [['tasmax', 'tasmin', 'pr'], ['elevation']],
    path_to_tifs: list = ['data/history/*.tif', 'data/elev/*.tif'],
    wind_in_box_file: str = 'wind_in_box.txt',
    half_side_size: int = 3,
    dset_num: int = 0,
    start_date: str = '2006-01-01',
    end_date: str = '2016-01-01',
    weatherstation_file: str = 'data_meteo_kk.csv',
    weatherstation_list: str = 'weatherstation_list.csv',
    station_names: list = ['Анапа', 'Армавир', 'Краснодар, Круглик', 'Сочи', 'Туапсе', 'Приморско-Ахтарск', 'Красная Поляна'],
    speed_th: float = 20,
    path_to_dump: str = os.path.join('data','nn_data')
):
    """
    Generate data for NN training
    """
    loaded_arr = np.loadtxt(wind_in_box_file)  
    # This loadedArr is a 2D array, therefore, we need to convert it to the original array shape.reshaping to get original matrice with original shape.
    cmip = loaded_arr.reshape(3652, 15, 22)
    blocks = make_blocks(feature_names, path_to_tifs, cmip=cmip, half_side_size = half_side_size, dset_num = dset_num)
    df = pd.read_csv(weatherstation_file)
    station_list = pd.read_csv(weatherstation_list)
    stations_pixs = get_pixel_stations(path_to_tifs[0], feature_names[0], station_names, station_list)
    target = get_y(df, start_date, end_date, speed_th=speed_th)
    X, y = assemble_numpy_ds(blocks, target, stations_pixs)
    trg_path = os.path.join(path_to_dump, 'target')
    obj_path = os.path.join(path_to_dump, 'objects')
    for k in X.keys():
        X_station = X[k]
        y_station = y[k]

        st_path = os.path.join(path_to_dump, k)
        if not os.path.isdir(st_path):
            os.makedirs(st_path)
        
        with open(os.path.join(st_path, 'objects.npy'),'wb') as f:
            # pickle.dump(X_station, f)
            np.save(f, X_station)
        with open(os.path.join(st_path, 'target.npy'),'wb') as f:
            # pickle.dump(y_station, f)
            np.save(f, y_station)


def generate_dataset_for_inference(
    feature_names: list = [['tasmax', 'tasmin', 'pr'], ['elevation']],
    path_to_tifs: list = ['data/history/*.tif', 'data/elev/*.tif'],
    wind_in_box_file: str = 'wind_in_box.txt',
    half_side_size: int = 3,
    dset_num: int = 0,
    start_date: str = '2006-01-01',
    end_date: str = '2016-01-01',
    weatherstation_file: str = 'data_meteo_kk.csv',
    weatherstation_list: str = 'weatherstation_list.csv',
    station_names: list = ['Анапа', 'Армавир', 'Краснодар, Круглик', 'Сочи', 'Туапсе', 'Приморско-Ахтарск', 'Красная Поляна'],
    speed_th: float = 20,
    path_to_dump: str = os.path.join('data','nn_data_grid_inference')

):
    """
    Generate data for NN inference
    """
    loaded_arr = np.loadtxt(wind_in_box_file)  
    # This loadedArr is a 2D array, therefore, we need to convert it to the original array shape.reshaping to get original matrice with original shape.
    cmip = loaded_arr.reshape(3652, 15, 22)
    blocks = make_blocks(feature_names, path_to_tifs, cmip=cmip, half_side_size = half_side_size, dset_num = dset_num)
    df = pd.read_csv(weatherstation_file)
    station_list = pd.read_csv(weatherstation_list)
    stations_pixs = get_pixel_stations(path_to_tifs[0], feature_names[0], station_names, station_list)
    target = get_y(df, start_date, end_date, speed_th=speed_th)
    X = assemble_numpy_ds(blocks=blocks, target=target, stations_pixs=stations_pixs, include_target=False)
    obj_path = os.path.join(path_to_dump, 'objects')
    for k in X.keys():
        X_station = X[k]

        st_path = os.path.join(path_to_dump, str(k))
        if not os.path.isdir(st_path):
            os.makedirs(st_path)
        
        with open(os.path.join(st_path, 'objects.npy'),'wb') as f:
            # pickle.dump(X_station, f)
            np.save(f, X_station)


def conv(
    path_to_data: str = os.path.join('data', 'nn_data'),
    train_part: float = 0.5,
    val_part: float = 0.25,
    test_part: float = 0.25,
    path_to_dump: str = os.path.join('data','nn_data'),
    batch_size: int = 1024,
    max_epochs: int = 500,
    gpus: str = '1',
    benchmark: bool = True,
    check_val_every_n_epoch: int = 1,
    path_to_cofig: str = os.path.join('../conf', 'conv_config.json')
):
    """
    Train NN 
    """
    st_split_dict = train_val_test_split(path_to_data, train = train_part, val = val_part, test = test_part, verbose = True)
    X, y = extract_splitted_data(path_to_dump, st_split_dict)
    with open(path_to_cofig) as fs:
        args = json.load(fs)
    logger = TensorBoardLogger(save_dir='logs/wind', name='windnet')
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_auroc", min_delta=0.00, patience=5, verbose=False, mode="max")
    trainer = pl.Trainer(max_epochs=max_epochs,
                        gpus=gpus,
                        benchmark=benchmark,
                        check_val_every_n_epoch=check_val_every_n_epoch,
                        callbacks=[early_stop_callback]
    )


    dm = WindDataModule(X=X, y=y, batch_size=batch_size, downsample=False)
    model = WindNetPL(args)

    trainer.fit(model, dm)

    trainer.test(model, dm)


def nn_inference(
    path_to_data: str = os.path.join('data', 'nn_data'),
    train_part: float = 0.5,
    val_part: float = 0.25,
    test_part: float = 0.25,
    path_to_dump: str = os.path.join('data','nn_data'),
    batch_size: int = 1024,
    path_to_cofig: str = os.path.join('../conf', 'conv_config.json'),
    max_epochs: int = 50,
    gpus: str = '1',
    benchmark: bool = True,
    check_val_every_n_epoch: int = 1,
    chk_path: str = "lightning_logs/version_0",
    features: list = ['elev', 'pr', 'tasmax', 'tasmin', 'wind'],
    path_to_inference_data: str = os.path.join('data', 'nn_data_grid_inference'),
    start_date = pd.to_datetime('2006-01-01'),
    exp_smooth = False,
    plot_interval: int = 100,
    plot_result: bool = False,
    plot_map: bool = False
):
    """
     Get NN's inference;
     It will generate a set of images, combine into .gif and .h264 video map
    """
    st_split_dict = train_val_test_split(path_to_data, train = train_part, val = val_part, test = test_part, verbose = True)
    X, y = extract_splitted_data(path_to_dump, st_split_dict)
    with open(path_to_cofig) as fs:
        args = json.load(fs)
    logger = TensorBoardLogger(save_dir='logs/wind', name='windnet')
    trainer = pl.Trainer(max_epochs=max_epochs,
                        gpus=gpus,
                        benchmark=benchmark,
                        check_val_every_n_epoch=check_val_every_n_epoch,
    )


    dm = WindDataModule(X=X, y=y, batch_size=batch_size, downsample=False)
    model = WindNetPL(args)

    # chk_path = "./lightning_logs/version_13/checkpoints/epoch=35-step=288.ckpt"
    chk_path = os.path.join(chk_path, "checkpoints", os.listdir(os.path.join(chk_path, "checkpoints"))[0])
    model2 = WindNetPL.load_from_checkpoint(chk_path, args=args)
    model2.eval()

    trainer.test(model2, dm)

    batch = next(iter(dm.test_dataloader()))
    images, _ = batch

    background = images[:100]
    test_images = images[100:103]

    e = shap.DeepExplainer(model2, background)
    shap_values = e.shap_values(test_images)

    for i in range(len(features)):
        print(features[i], abs((torch.mean(torch.tensor(shap_values[0]), dim=[0, 2, 3])[i]).numpy()) + (torch.mean(torch.tensor(shap_values[1]), dim=[0, 2, 3])[i]).numpy() )

    
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
    shap.image_plot(shap_numpy, -test_numpy)

    # path_to_data = os.path.join('data', 'nn_data_2021')
    grid_inference_ = {}

    for i, curr_pix in tqdm(enumerate(os.listdir(path_to_inference_data))):
        X, y = extract_splitted_data(path_to_inference_data, {'Grid': [curr_pix]})
        X_grid_t = torch.tensor(X["Grid"], device=model2.device).double()
        inference = model2(dm.transform(X_grid_t)).detach().cpu().numpy()
        grid_inference_[curr_pix] = inference   
    
    grid_inference = {}
    for k in grid_inference_.keys():
        try:
            k1, k2 = k[1:-1].split(', ')
            k1 = int(k1)
            
            k2 = int(k2)
            grid_inference[(k1, k2)] = grid_inference_[k]
        except AttributeError:
            pass
    del grid_inference_

    keys = list(grid_inference.keys())
    sorted_tmp = sorted(keys, key=lambda x: x[0])
    shift_x = sorted_tmp[0][0] + 1
    max_x = sorted_tmp[-1][0] + shift_x
    sorted_tmp = sorted(keys, key=lambda x: x[1])
    shift_y = sorted_tmp[0][1] + 1
    max_y = sorted_tmp[-1][1] + shift_y
    some_key = keys[0]
    grid = np.zeros((grid_inference[some_key].shape[0], max_x, max_y))

    for k in grid_inference.keys():
        grid[:, k[0], k[1]] = grid_inference[k][:, 1]
    
    # start_date = pd.to_datetime('2021-01-01')

    if exp_smooth:
        grid_new = np.zeros_like(grid)
        for i in range(grid.shape[0]):
            grid_new[i,:,:] = ewma_vectorized_2d(data=grid[i,:,:], alpha=0.7, axis=1)
            grid_new[i,:,:] = ewma_vectorized_2d(data=grid_new[i,:,:], alpha=0.7, axis=0)
        else:
            grid_new = grid

    if plot_result:
        X = np.arange(36.5, 42.0, 0.5)
        Y = np.arange(43.5, 47.5, 0.5)[::-1]

        for t in list(range(grid_new.shape[0]))[::plot_interval]:
            data = grid_new[t, :, :]
            fig = plt.figure(figsize=(7, 3.7))

            ax = fig.add_subplot(111)
            ax.set_title('t = ' + str(start_date + pd.DateOffset(1) * t))
            plt.imshow(data,vmin=0, vmax=0.6)
            # plt.colorbar()
            ax.set_aspect('equal')
            ax.set_xticks(np.arange(0,grid_new.shape[2],2), labels=X)
            ax.set_yticks(np.arange(grid_new.shape[1]-1,-1,-2), labels=Y)

            cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
            cax.get_xaxis().set_visible(False)
            cax.get_yaxis().set_visible(False)
            cax.set_frame_on(False)
            plt.colorbar(orientation='vertical')
            plt.savefig('tmp_dump/' + str(t) + '.png')
            # plt.show()

        images = []
        filenames = [os.path.join('tmp_dump', str(f) + '.png') for f in list(range(grid.shape[0]))[::plot_interval]]
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('tmp_dump/wind_prob_maps.gif', images, duration=0.5)

    if plot_map:
        X_full = np.arange(36.5, 42, 0.25)
        Y_full = np.arange(43.5, 47.25, 0.25)
    
        df = map_to_pandas(grid=grid_new, x_axis=X_full, y_axis=Y_full, start_date=start_date, day_interval=plot_interval)

        column = 'value'

        vmin = 0
        vmax = 0.6

        name = 0
        if not os.path.exists('anim_future'):
            os.makedirs('anim_future')
        for t in list(range(grid_new.shape[0]))[::plot_interval]:
            date = str(start_date + pd.DateOffset(1) * t)
            plot_map(df[df['date'] == date], column, part_world_to_plot='KK', vmin=vmin, vmax=vmax, 
                    img_path=f'anim_future/{name}.jpg', text = date,
                    show=False)
            name+=1
            plt.close()
