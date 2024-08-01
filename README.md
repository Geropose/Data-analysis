# Data-analysis

Descripción
Este proyecto tiene como objetivo analizar un conjunto de datos de hongos para determinar si un hongo es venenoso, estableciendo relaciones entre los atributos del conjunto de datos. El análisis se realiza utilizando la biblioteca mlxtend para aplicar el algoritmo apriori y las reglas de asociación.

Contenido del Archivo
El archivo Asociacion2023_Dataset.py contiene los siguientes componentes:

Carga del Dataset:

Se utiliza la biblioteca openml para cargar el dataset número 24.
El dataset describe 23 especies de hongos en base a características físicas y si son comestibles o venenosos.
Preprocesamiento:

Se utiliza get_dummies para generar la representación one-hot-encoded del dataset.
Aplicación del algoritmo apriori con un soporte mínimo de 0.3.
Filtrado de itemsets frecuentes con longitud menor o igual a tres.
Reglas de Asociación:

Establecimiento del parámetro de confianza en 0.9.
Filtrado de reglas donde el consecuente es la clase venenosa.
Dos criterios de filtrado adicionales:
Conviction menor a 1.
Antecedentes específicos y lift mayor a 1.01.
