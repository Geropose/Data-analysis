# -*- coding: utf-8 -*-
"""
Cargamos la librería que nos permite acceder al dataset
"""

# soporte para cargar dataset de https://www.openml.org/
!pip install openml
import openml

"""Accedemos al dataset"""

import pandas as pd

# se debe indicar aquí cual es el dataset que han elegido de OPENML
dataset = openml.datasets.get_dataset(43009)

# separamos las información almacenada en el dataset
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='dataframe',
    target=dataset.default_target_attribute
)
X

"""
el objetivo de aplicar tecnicas de clasificación sobre un conjunto de datos es poder organizar y categorizar los datos en diferentes clases, creando un modelo que nos permita clasificar nuevos datos, es decir dado un modelo poder predecir a que clase pertenece el nuevo dato.
Se establecio como objetivo determinar si un individuo es donante de sangre (1.0) o no lo es (0.0), se tomo como variable dependiente la columna Blood_Donor.

fue necesario realizar preprocesamiento.
Primero se utilizo OneHotEncoder para convertir en formato OneHot las columnas 'Sex': 'm' y 'f'; 'Category': '0=Blood Donor', '0s= suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis'; las cuales son completadas con 1 y 0.
Además se utilizo SimpleImputer para completar los valores faltantes, utilizando el valor -1.
Se eliminaron también columnas que para nuestro criterio no aportaban información relevante para nuestra clasificación.


"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')

datos = enc.fit_transform(X.Sex.values.reshape(-1, 1)).toarray()
X = pd.concat([X.drop("Sex", axis = 1), pd.DataFrame(datos, columns=enc.categories_[0])], axis = 1)

datos2 = enc.fit_transform(X.Category.values.reshape(-1, 1)).toarray()
X = pd.concat([X.drop("Category", axis = 1), pd.DataFrame(datos2, columns=enc.categories_[0])], axis = 1)

imp_mean = SimpleImputer(strategy="constant", fill_value=-1)

Y = imp_mean.fit_transform(X)
dset = pd.DataFrame(Y, columns= X.columns)


dset.rename(columns= {'2=Fibrosis': 'Fibrosis', '0=Blood Donor': 'Blood_Donor', '0s=suspect Blood Donor': 'suspectBlood_Donor', '1=Hepatitis':'Hepatitis', '3=Cirrhosis':'Cirrhosis'},  inplace = True)
dset = dset.drop(['Unnamed: 0'], axis= 1)
dset = dset.drop(['Hepatitis'], axis= 1)
dset = dset.drop(['Fibrosis'], axis= 1)
dset = dset.drop(['Cirrhosis'], axis= 1)
dset = dset.drop(['suspectBlood_Donor'], axis= 1)

y = dset.Blood_Donor
dset = dset.drop(['Blood_Donor'], axis= 1)

dset

"""Elegimos las técnicas Arbol de decisión y Vecinos más cercanos, estos clasificadores se evaluaron con la técnica hold-out ya que se tienen los datos suficientes para particionar el dataSet en dos conjuntos de datos (Trainning y Testing)."""

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# split del dataset en training (70%) y testing (30%)
X_train, X_test, y_train, y_test = train_test_split(dset, y, test_size=0.3)

##TECNICA 1 DE CLASIFICACIÓN
# entrena el clasificador Arbol de Decision
clf_t = tree.DecisionTreeClassifier()
clf_t.fit(X_train, y_train)

# evalua el clasificador Arbol de Decision
clf_t.score(X_test, y_test)

from sklearn.neighbors import KNeighborsClassifier

##TECNICA 2 DE CLASIFICACIÓN
## entrena el clasificador Vecinos más cercanos
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)

# evalua el clasificador Vecinos más cercanos
clf_knn.score(X_test, y_test)

"""
Se decidio utilizar la métrica f1-score para comparar los clasificadores ya que calcula en base a la precisión y al recall. Viendo los resultados obtenidos en el siguiente código se visualiza que las dos alternativas ofrecen resultados similares pero por una minima diferencia el árbol de decisión clasifica mejor que vecinos más cercanos por ende sería la que recomendariamos utilizar.

"""

from sklearn.metrics import classification_report

## Análisis métrico de la alternativa 1 - Arboles de decisión
predictions_tree = clf_t.predict(X_test)
print(classification_report(y_test, predictions_tree, target_names= ["Blood donor", "No blood donor"]))

## Análisis métrico de la alternativa 2 - Vecinos más cercanos
predictions_knn = clf_knn.predict(X_test)
print(classification_report(y_test, predictions_knn, target_names= ["Blood donor", "No blood donor"]))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

##VISUALIZACIÓN DEL ARBOL
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
columns_names = dset.columns.values

fig, ax = plt.subplots(figsize=(20, 20))
plot_tree(clf, feature_names =columns_names, class_names=["Blood donor", "No blood donor"], ax=ax)
plt.show()

