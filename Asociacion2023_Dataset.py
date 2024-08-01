# -*- coding: utf-8 -*-
"""
Cargamos la librería que nos permite acceder al dataset
"""

# soporte para cargar dataset de https://www.openml.org/
!pip install openml
import openml

"""Accedemos al dataset"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

# indicamos cual dataset queremos utilizar, en este caso el nro. 24
dataset = openml.datasets.get_dataset(24)

# separamos las información almacenada en el dataset
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='dataframe',
    target=dataset.default_target_attribute
)

#  concatenamos la información relevante en un único DataFrame
df = pd.concat([X, y], axis=1)
df

"""el dataset indicado para trabajar describe un conjunto de 23 especies de hongos en base a sus distintas caracteristicas fisicas, algunas de ellas son: superficie, forma, color, entre otras. Por otro lado, otra gran diferenciación que se realiza en el dataset es si los hongos son comestibles o venenosos.

nuestro objetivo es determinar si un hongo es venenoso, estableciendo relaciones entre los atributos del conjunto de datos.
Los patrones que esperamos encontrar, son similitudes del tipo por ejemplo: color de sombrero marron, olor almendrado y mucho espaciado entre las branquias entonces el hongo es venenoso.

El pre-procesamiento se realizo aplicando la función get_dummies para generar la representación one-hot-encoded, y el algoritmo apriori con un soporte minimo de 0.3, ya que con un soporte mayor no obteniamos los datos necesarios para los patrones establecidos en el inciso B.
Realizamos además un filtrado que solo contemplase los itemset frecuentes de longitud menor o igual a tres en relación al objetivo establecido en el inciso B.
"""

#from mlxtend.frequent_patterns import apriori

#df_brand_dummies = pd.get_dummies(df, columns = df.columns)
#apriori(df_brand_dummies, min_support = 0.3, use_colnames = True)

#frequent_itemsets = apriori(df_brand_dummies, min_support=0.3, use_colnames=True)
#frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

#frequent_itemsets[ (frequent_itemsets['length'] <= 3)]

from mlxtend.frequent_patterns import apriori

df_brand_dummies = pd.get_dummies(df, columns = df.columns)
apriori(df_brand_dummies, min_support = 0.3, use_colnames = True)

frequent_itemsets = apriori(df_brand_dummies, min_support=0.3, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets

"""
se establecio el parámetro de confianza en 0.9, como métrica, de manera de asegurarnos de tener datos lo suficientemente confiables para lograr visualizar nuestro objetivo.
"""

from mlxtend.frequent_patterns import association_rules

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)

print("Se imprimen las reglas extraidas con los criterios ingresados\n")
rules

"""El primer filtrado se realiza teniendo en cuenta que el consecuente sea clase venenoso, esto se visualiza a continuación:"""

##rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))

#En base al objetivo determinado en el inciso A, buscamos los consecuentes que retornan
# hongos comestibles

##RE-ENTREGA
rules = rules[rules['consequents'].apply(lambda x: 'class_e' in x)]
rules

"""Además como un segundo filtrado para lograr resultados más especificos establecimos antecedentes particulares:

**Criterios de filtrado:
En el filtro número 1 se utiliza la métrica conviction menores a 1 ya que cuando conviction tiende a cero significa que la regla es una casualidad. El resultado es vacío, entonces ninguna regla es casual.
En el filtro número 2 se realiza un filtrado en base a dos antecedentes que en el inciso D aparecen con mayor frecuencia, con estos antecedentes utilizamos la métrica lift que siendo mayor a 1 indica que los antecedentes tienen una relación positva o asociación fuerte. Obtenemos 134 filas de reglas con este filtro.**
"""

## RE-ENTREGA

print('Filtro numero 1')
print(rules[(rules['conviction'] < 1.0) & (rules['consequents'] == {'class_e'})])

print('Filtro numero 2')
print(rules[(rules['antecedents'].apply(lambda x: "odor_n" in x)) & (rules['antecedents'].apply(lambda x: "gill-attachment_f" in x)) & (rules['lift'] > 1.01) & (rules['consequents'] == {'class_e'})])

#rules

"""
En el inciso E obteniamos 134 filas con el filtrado número 2 y al aplicar el filtro obtenemos como resultado 48 filas donde en todas las reglas aparecen los antecedentes 'odor_n' y 'gill-attachment_f'.
Utilizando este filtro podemos afirmar que no tenemos ninguna regla redundante ya que el conjunto de datos sigue el patrón:
odor_n, gill-attachment_f -> class_e**
"""

#rules = rules[rules['antecedents'].apply(lambda x: 'veil-color_w' in x)]
#rules = rules[rules['antecedents'].apply(lambda x: 'gill-spacing_c' in x)]
#rules

rules [rules['antecedents'].apply(lambda x: "odor_n" in x) & (rules['antecedents'].apply(lambda x: "gill-attachment_f" in x)) & (rules['lift'] > 1.01) & (rules['consequents'] == {'class_e'})]
#rules = association_rules.prune(rules, "antecedents", "consequents")

"""según las reglas obtenidas por los filtrados en las tablas anteriores, podemos deducir que la mayoria de las reglas contienen los atributos gill-spacing_c, veil-color_w y estos implican hongos de clase venenosa. Podemos definir entonces la regla: (gill-spacing_c, veil-color_w) -> class-p

En las páginas web de los parques nacionales o de lugares de la naturaleza abiertos a todo publico, se podrían colocar infografías que indiquen que si un hongo tiene poco espacio entre las branquias y un velo de color blanco probablemente este sea venenoso por lo cual no es recomendable agarrarlo, consumirlo, etc.

"""