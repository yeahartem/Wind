# Environments' purposes
* `pygdal` (`environment_pygdal.yml`): `generate_dataset.ipynb`, `generate_dataset_for_inference.ipynb` -- work with `gdal` library (dataset extraction from .tif files), works only with python 3.5
* `kl-cpd-copy` (`environment_torch.yml`): `conv.ipynb`, `nn_inference.ipynb` -- training NN, inferring
* `ee` (`environment_ee.yml`): work with Earth Engine via python API
# Data
* Run `generate_dataset.ipynb` to generate data for NN training under `pygdal` env
* Run `generate_dataset_for_inference.ipynb` to generate data for NN inference under `pygdal` env
# NN training
* Run `conv.ipynb` to train NN under `kl-cpd-copy` env
* Run `nn_inference.ipynb` to get NN's inference under `kl-cpd-copy` env; it will generate a set of images, combine into .gif and .h264 video map
