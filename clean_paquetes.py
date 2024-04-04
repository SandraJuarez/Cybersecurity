import torch
from sklearn.preprocessing import MinMaxScaler
import copy
import numpy as np
import pandas as pd

from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split

from torch import nn, optim

import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder


def create_sequences(data, sequence_length):
    sequences = []
    #columna_close=3
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length,:]
        sequences.append(seq)
    return sequences

# Create sequences and corresponding labels for training set


def create_dataset(df):

  sequences = df.astype(np.float32).to_numpy().tolist()

  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  datasetT=torch.stack(dataset)
  datasetS=torch.stack(create_sequences(datasetT,10))
  print('el shape del dataset',datasetS.shape)
  n_seq=datasetS.shape[0]
  seq_len=datasetS.shape[1]
  n_features = datasetS.shape[2]
  
  
  print(n_seq, seq_len, n_features)
  

  return dataset, seq_len, n_features


def clean_and_get_Data(df,seq_len):
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    #data = pd.read_csv(filename)
    #df = pd.DataFrame(data)
    # primero, convertimos la columna 'Source' a tipo de datos de cadena (str)
    df['Source'] = df['Source'].str.replace('.', '')
    df['Destination'] = df['Destination'].str.replace('.', '')
    df['Source'] = df['Source'].astype(str)
    df['Destination'] = df['Destination'].astype(str)
    

    # Usamos expresiones regulares para encontrar solo las entradas que contienen solo números
    df = df[df['Source'].str.match(r'^\d+$')]
    df = df[df['Destination'].str.match(r'^\d+$')]

    unique_protocol_values = df['Protocol'].unique()

    # Crear un mapeo basado en la lista de valores únicos
    protocol_mapping = {protocol: index for index, protocol in enumerate(unique_protocol_values, start=1)}

    # Reemplazar las siglas de protocolos con los índices correspondientes
    df['Protocol'] = df['Protocol'].map(protocol_mapping)

    df = df.drop(columns=['Info','No.','Source','Destination'])
    # Suponiendo que 'df' es tu DataFrame y 'columns_to_normalize' es una lista de columnas a normalizar
    columns_to_normalize = ['Time','Length','Protocol']  # Lista de columnas a normalizar
    #df = df.drop(columns=[])

    scaler = MinMaxScaler()  # Inicializar el MinMaxScaler

    # Normalizar las columnas seleccionadas
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    #And we save the clean dataset
    file_name = 'clean_captures_data.csv'
    df.to_csv(file_name, index=False)

    datas=df.iloc[:, :].values
    print(datas.shape)
    sequences = create_sequences(datas, seq_len)
    #print('Sequences',len(sequences))
    
    train_df, val_df = train_test_split(
    df,
    test_size=1,
    random_state=RANDOM_SEED
    )


    #train_size = int(0.8 * len(sequences))
    
    #train_data = sequences[:train_size]
    #test_data = sequences[train_size:]
    #print(train_data.shape)
    # Crear DataLoader para cargar los datos
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    train_dataset, seq_len, n_features= create_dataset(train_df)
    #seq_len=25
    #n_features=5
    return train_dataset,seq_len,n_features
#test_anomaly_dataset, _, _ = create_dataset(anomaly_df)


if __name__=="__MAIN__":
   clean_and_get_Data()