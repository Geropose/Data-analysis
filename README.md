# Data-analysis

## Descripción del archivo Asociacion2023_Dataset
Este proyecto analiza un conjunto de datos de hongos para determinar si un hongo es venenoso, estableciendo relaciones entre los atributos del conjunto de datos. El análisis se realiza utilizando la biblioteca mlxtend para aplicar el algoritmo Apriori y las reglas       de asociación.

### Contenido del Archivo Asociacion2023_Dataset.py:
- **Carga del Dataset**:Se utiliza la biblioteca openml para cargar el dataset número 24, que describe 23 especies de hongos en base a características físicas y si son comestibles o venenosos.
- **Preprocesamiento**:
- Se utiliza get_dummies para generar la representación one-hot-encoded del dataset.
- Se aplica el algoritmo Apriori con un soporte mínimo de 0.3.
- Filtrado de itemsets frecuentes con longitud menor o igual a tres.
- **Reglas de Asociación**:
-Establecimiento del parámetro de confianza en 0.9.
-Filtrado de reglas donde el consecuente es la clase venenosa.
-Dos criterios de filtrado adicionales:
-Conviction menor a 1.
-Antecedentes específicos y lift mayor a 1.01.

## Descripción del archivo Clasificacion2023_Dataset
Este proyecto aplica técnicas de clasificación sobre un conjunto de datos para predecir si un individuo es donante de sangre (1.0) o no lo es (0.0). Se utilizan clasificadores como el Árbol de Decisión y Vecinos más Cercanos, y se evalúan utilizando la técnica hold-     out.
### Contenido del Archivo Clasificacion2023_Dataset.py:
- **Carga del Dataset**Se utiliza la biblioteca openml para cargar el dataset número 43009.
- **Preprocesamiento**:
-Utilización de OneHotEncoder para convertir columnas categóricas a formato OneHot.
-Utilización de SimpleImputer para completar valores faltantes.
-Eliminación de columnas irrelevantes para la clasificación.
- **Técnicas de Clasificación**:   
-Clasificador Árbol de Decisión.
-Clasificador Vecinos más Cercanos.
-Evaluación de los clasificadores con la métrica f1-score.
