import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import data_mining as dm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import pickle



def pca(data, labels):
    #use pca for comparison with the tsne
    # Inicializar PCA
    pca = PCA(n_components=2)
    # Ajustar y transformar los datos
    reduced_X_pca = pca.fit_transform(data)

    #save the pca model
    filename = 'pca_model.sav'
    pickle.dump(pca, open(filename, 'wb'))


    #mlp classifier with pca
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(reduced_X_pca, labels, test_size=0.2, random_state=42)

    # Inicializar y entrenar un clasificador de Regresión Logística
    classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    classifier.fit(X_train, y_train)

    # Predecir las etiquetas de clase para los datos de prueba
    y_pred = classifier.predict(X_test)

    # Calcular la precisión del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del clasificador MLP:", accuracy)  

    #save the mlp classifier
    filename = 'mlp_classifier.sav'
    pickle.dump(classifier, open(filename, 'wb'))


def prediction(datos):
    #load the pca model
    filename = 'pca_model.sav'
    pca = pickle.load(open(filename, 'rb'))

    #transform the data
    datos = pca.transform(datos)

    #Load the model
    filename = 'mlp_classifier.sav'
    model = pickle.load(open(filename, 'rb'))
    #predict the data
    prediction = model.predict(datos)

    return prediction
