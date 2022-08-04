import pandas as pd
import json
import os
import sys
import numpy as np


def processing(v):
  if type(v) == float or type(v) == int:
        return v
  elif v.strip() == '':
        return np.nan
  else:
        return float(v.strip())


def weatherstations_loader(
    path_to_data: str = 'data_meteo_full/sroks',
    path_to_station_dict: str = 'data_meteo_full/station_dict_full.json',
    to_save: bool = False,
    path_for_saving: str = 'data_meteo_full/data_meteo_full.csv'  
):
    with open(path_to_station_dict, "r") as my_file:
        d_json = my_file.read()
    d = json.loads(d_json)
    stations_dict = dict({})

    dir_list = os.listdir(path_to_data)
    for dir in dir_list:
        if dir[:2] == 'wr':
            file_list = os.listdir(path_to_data + '/' + dir)
            for file in file_list:
                if len(file[:-4]) == 5 and file[-3:] == 'txt':
                    Station = pd.read_table(path_to_data + '/' + dir + '/' + file, sep=';', header=None)
                    print(path_to_data + '/' + dir + '/' + file)
                    Station_date = pd.DataFrame({'year': Station.iloc[:,1],
                                                'month': Station.iloc[:,2],
                                                'day': Station.iloc[:,3]})
                    # Data_Station = pd.DataFrame(np.zeros((12760, 13)))
                    Data_Station = pd.DataFrame({'Дата': pd.to_datetime(Station_date)})
                    Data_Station = Data_Station.join(pd.DataFrame({'Название метеостанции': [d[file[:-3]+'0'] for i in range(len(Station))]}))
                    Data_Station = Data_Station.join(pd.DataFrame({'Максимальная скорость': Station.iloc[:,27]}))        
                    Data_Station = Data_Station.join(pd.DataFrame({'Средняя скорость ветра': Station.iloc[:,26]}))        
                    Data_Station = Data_Station.join(pd.DataFrame({'Направление ветра': Station.iloc[:,25]}))       
                    Data_Station = Data_Station.join(pd.DataFrame({'Температура воздуха по сухому терм-ру': Station.iloc[:,34]}))          
                    Data_Station = Data_Station.join(pd.DataFrame({'Атмосферное давление на уровне станции': Station.iloc[:,44]}))       
                    Data_Station = Data_Station.join(pd.DataFrame({'Атмосферное давление на уровне моря': Station.iloc[:,45]}))         
                    Data_Station = Data_Station.join(pd.DataFrame({'Сумма осадков': Station.iloc[:,28]}))        
                    Data_Station = Data_Station.join(pd.DataFrame({'Температура поверхности почвы': Station.iloc[:,29]}))         
                    Data_Station = Data_Station.join(pd.DataFrame({'Парциальное давление водяного пара': Station.iloc[:,40]}))         
                    Data_Station = Data_Station.join(pd.DataFrame({'Относительная влажность воздуха': Station.iloc[:,41]}))         
                    Data_Station = Data_Station.join(pd.DataFrame({'Температура точки росы': Station.iloc[:,43]}))      
                    for i in range(2,13):
                        Data_Station.iloc[:,i] = Data_Station.iloc[:,i].apply(processing)
                    stations_dict[d[file[:-3]+'0']] = Data_Station
                    
    data_meteo_full = pd.concat(stations_dict.values())

    if to_save:
        data_meteo_full.to_csv(path_for_saving, index=False)

    return data_meteo_full