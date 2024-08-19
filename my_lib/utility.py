import numpy as np


def normalizar(data, a=0, b=1):
    # Funcion para normalizar una población de valores en el rango [a,b]recibe como entrada:
    # - data (numpy array): una lista de valores o población
    # -a (float): El límite inferior del rango de normalización utilizado (por defecto 0).
    # -b (float): El límite superior del rango de normalización utilizado (por defecto 1).
    # retorna:
    # xnomr(numpy array): el arreglo con los numeros normalizados en el ranbo [a,b]

    xmin = np.min(data)
    xmax = np.max(data)

    if np.all(xmin == xmax):
        return np.full_like(data, xmin)

    xnorm = a + ((data - xmin) / (xmax - xmin)) * (b - a)

    return xnorm


def desnormalizar(norm_data, data, a=0, b=1):
    # Funcion para desnomarlizar datos normalizados a su rango orinial, recibe como entrada:
    # - norm_data (float): los datos normalizados
    # - data (numpy array): el conjunto de datos originales
    # - a (float): El límite inferior del rango de normalización utilizado (por defecto 0).
    # retorna:
    # - x(int): Los datos desnormalizados a su rango original

    xmin = np.min(data)
    xmax = np.max(data)

    x = xmin + (xmax - xmin) * ((norm_data - a) / (b - a))

    return x


def binarizar(nbites, datos):
    """
    Convierte números decimales en su representación binaria utilizando un número específico de bits.

    Parámetros de entrada:
    - nbites (int): Número de bits para representar los números binarios.
    - datos (numpy array): Arreglo de números decimales a convertir en su representación binaria.

    Retorna:
    - numpy string array: Arreglo de cadenas que representan los números en formato binario, donde cada número está ajustado a `nbites` bits.
    """
    datos = datos.astype(np.int64)
    binary_repr_v = np.vectorize(np.binary_repr)
    return binary_repr_v(datos, nbites)


def bin_to_dec(datos):
    """
    Convierte un numero binario representado con cadenas de texto a un numero decimal entero

    Parametros de entrada:
    - datos (string): los numeros en binario que seran convertidos a su representacion decimal

    Retorna:
    - numpy array : Arreglo de numeros enteros decimales
    """

    datos_dec = []

    return


import numpy as np


def generar_poblacion(datos, ndatos):
    """
    Genera una población seleccionando aleatoriamente un subconjunto de elementos de los datos originales.

    Parámetros:
    - datos (iterable o numpy array): Conjunto de datos del cual se seleccionarán elementos.
    - ndatos (int): Número de elementos que se seleccionarán aleatoriamente para formar la población.

    Retorna:
    - numpy array: Arreglo de `ndatos` elementos seleccionados aleatoriamente del conjunto original `datos`.
    """

    n = len(datos)
    indices = np.random.choice(range(n), ndatos, replace=False)
    poblacion = np.array([datos[i] for i in indices])

    return poblacion
