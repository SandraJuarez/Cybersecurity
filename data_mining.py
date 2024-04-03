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

def create_dataset(df):

  sequences = df.astype(np.float32).to_numpy().tolist()

  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]

  n_seq, seq_len, n_features = torch.stack(dataset).shape

  return dataset, seq_len, n_features


def clean_and_get_Data(filename):
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    data = pd.read_csv(filename)
    df = pd.DataFrame(data)
    # primero, convertimos la columna 'Source' a tipo de datos de cadena (str)
    df['Source'] = df['Source'].str.replace('.', '')
    df['Destination'] = df['Destination'].str.replace('.', '')
    df['Source'] = df['Source'].astype(str)
    df['Destination'] = df['Destination'].astype(str)
    

    # Usamos expresiones regulares para encontrar solo las entradas que contienen solo números
    df = df[df['Source'].str.match(r'^\d+$')]
    df = df[df['Destination'].str.match(r'^\d+$')]

    '''
    protocol_mapping = {
        'TCP': 1,
        'UDP': 2,
        'HTTP': 3,
        'HTTPS': 4,
        'FTP': 5,
        'SMTP': 6,
        'POP3': 7,
        'IMAP': 8,
        'SNMP': 9,
        'DHCP': 10,
        'DNS': 11,
        'ICMP':12,
        'ARP':13,
        'SSDP':14,
        'TLSv1.2':15,
        'TLSv1.3':15,
        'MDNS':16,
        'ICMPv6':17,
        'SSL':18,
        'SIGCOMP':19,
        'CLASSIC-STU':20,
        'LLDP':21,

    }
    '''
    unique_protocol_values = df['Protocol'].unique()

    # Crear un mapeo basado en la lista de valores únicos
    protocol_mapping = {protocol: index for index, protocol in enumerate(unique_protocol_values, start=1)}

    # Reemplazar las siglas de protocolos con los índices correspondientes
    df['Protocol'] = df['Protocol'].map(protocol_mapping)

    df = df.drop(columns=['Info','No.'])
    # Suponiendo que 'df' es tu DataFrame y 'columns_to_normalize' es una lista de columnas a normalizar
    columns_to_normalize = ['Time', 'Source', 'Destination','Length']  # Lista de columnas a normalizar

    scaler = MinMaxScaler()  # Inicializar el MinMaxScaler

    # Normalizar las columnas seleccionadas
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    #And we save the clean dataset
    file_name = 'clean_dataset' + file_name + '.csv'
    df.to_csv(file_name, index=False)
    train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    random_state=RANDOM_SEED
    )

    val_df, test_df = train_test_split(
    df,
    test_size=0.33, 
    random_state=RANDOM_SEED
    )

    train_dataset, seq_len, n_features = create_dataset(train_df)
    val_dataset, _, _ = create_dataset(val_df)
    test_normal_dataset, _, _ = create_dataset(test_df)
    return train_dataset,seq_len,n_features,val_dataset,test_normal_dataset
#test_anomaly_dataset, _, _ = create_dataset(anomaly_df)


if __name__=="__MAIN__":
   clean_and_get_Data()
   




