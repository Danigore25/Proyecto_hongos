'''
NAME
    Hongos_clases.py
VERSION
    1.0
AUTHOR
    Daniela Goretti Castillo León <danigore22@gmail.com>
    José Rodelmar Ocampo Luna <joserodelmar@gmail.com>
DESCRIPTION
    Este programa busca un modelo con el algoritmo k-Neighbors para predecir si un hongo es venenoso o comestible de acuerdo a sus características.
CATEGORY
    Algoritmos de aprendizaje supervisado
USAGE
    Hongos_clases.py [sin opciones]
ARGUMENTS
    No se requieren argumentos.
PACKAGES
    import numpy as np
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_confusion_matrix
INPUT
     Data set Mushroom o datos de hongos con los 22 atributos.
OUTPUT
    Valores de presición del modelo, reporte de clasificación y matriz de confusión.
EXAMPLES
    Example 1: Se tiene el data set de hongos Mushroom con el que vamos a entrenar a nuestro modelo. Importamos las librerías descritas en la sección packages, utilizamos
    LabelEncoder para poder pasar las características de las muestras de hongos como una matriz. Leemos el data set con las características, guardamos las líneas del archivo
    con los datos y los guardamos en una lista. Guardamos también las clases de las muestras en una lista nueva (llamada ejemplos2). Realizamos un arreglo con dos dimensiones
    con las características, transformamos las muestras a valores numéricos y lo reacomodamos. Separamos el dataset en las fases de entrenamiento y evaluación poniendo un 
    porcentaje de 30% para la fase test. Preparamos el algoritmo k-Neighbors y vamos actualizando el modelo con el método .fit. Obtenemos los resultados para las medidas de
    evaluación del modelo, hacemos el reporte de clasificación y la matriz de confusión para analizar a nuestro modelo.
    
GITHUB LINK
    https://github.com/Danigore25/Proyecto_hongos/blob/main/Hongos_clases.py
'''


import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


# 1. Usar LabelEncoder
le = preprocessing.LabelEncoder()
# Cargar el dataset de hongos para cargar los arreglos
ejemplos = []
ejemplos2 = []
# Separacion del archivo agaricus-Lepidota.data (dentro de carpeta Hongos-20220226) en caracteristicas y clases por terminal UNIX
# Clases :
# cut -d, -f1 Hongos-20220226/agaricus-lepiota.data > data_clases.data
# Caracteristicas :
# cut -d, -f2-23 Hongos-20220226/agaricus-lepiota.data > data_char.data

# 2. Leer el dataset de caracteristicas, dividirlo por ejemplo y guardarlo en la lista de "ejemplos"
mdata = open("data_char.data", "r")
mdataset = mdata.readlines()
mdata.close()
for line in mdataset:
    final = line.find("\n',")
    prim_lin = line[:final]
    sequence_not = line.find("'", final+1)
    sequence = line[final+1:sequence_not]
    ejemplos.append(sequence)

# 3. Leer el dataset solo con clases, dividirlo por ejemplo y guardarlo en la lista "ejemplos2"
danam = open("data_clases.data", "r")
datanam = danam.readlines()
danam.close()
for line in datanam:
    final2 = line.find("\n',")
    prim_lin2 = line[:final2]
    sequence_not2 = line.find("'", final2+1)
    sequence2 = line[final2+1:sequence_not2]
    ejemplos2.append(sequence2)

# 4. Crear un arreglo 2d con los datos de caracteristicas
arr_2d = np.reshape(ejemplos, (8124,1))
# Comprobar el rearreglo
print(ejemplos[0:4])
print(arr_2d[0:5])
print(ejemplos2[0:4])
# Codificar o pasar las instancias cualitativas de caracteristicas a instancias numericos para poder usar el algoritmo
y2 = le.fit_transform(arr_2d)
# Volver a reacomodar para un arreglo 2d
y22 = np.reshape(y2, (8124,1))
# Corroborar
print(y22[0:23])

# 5. Leemos conjunto de ejemplos y los valores de clase para los mismos
X = y22
print(X[:5])
y = ejemplos2
print(y[:5])
# Separamos el dataset en dos: entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# 6. Clasificación K Nearest neighbors 
k = 5
# Definición del clasificador
classifier = KNeighborsClassifier(n_neighbors=k)
# Entrenamiento del clasificador con lo datos de entrenamiento y valores de clase para cada ejemplo
classifier.fit(X_train, y_train)
# Predicción con el clasificador entrenado en los datos de evaluación 
y_predict = classifier.predict(X_test)
y_predict

# 7. Comparar las medidas de rendimiento del clasificador
print("Accuracy: {}".format(accuracy_score(y_test, y_predict)))
print("Precision: {}".format(precision_score(y_test, y_predict, average="macro")))
print("Recall: {}".format(recall_score(y_test, y_predict, average="macro")))
print("F-score: {}".format(f1_score(y_test, y_predict, average="macro")))

# 8. Hacer el reporte de clasificación
target_names = ['Poisonous', 'Edible']
print(classification_report(y_test, y_predict, target_names=target_names))

# 9. Hacer la matriz de confusión
print(confusion_matrix(y_test, y_predict))
plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues, display_labels=['Poisonous', 'Edible'])  
