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


import numpy as np


def bin_to_dec(datos):
    """
    Convierte números binarios representados como cadenas de texto a su correspondiente número decimal entero.

    Parámetros de entrada:
    - datos (string, list o numpy.ndarray): Si es una cadena, debe ser un número binario representado como texto.
    Si es una lista, debe contener números binarios como cadenas.

    Retorna:
    - numpy array: Arreglo de números enteros decimales correspondientes a los valores binarios.
    Si se proporciona una sola cadena binaria, se devuelve un único número decimal.
    """
    # Verificar si 'datos' es un numpy.ndarray, lista o cadena
    if isinstance(datos, np.ndarray) or isinstance(datos, list):
        # Convertir cada número binario a su representación decimal
        poblacion_dec = [int(i, 2) for i in datos]
        return np.array(poblacion_dec)
    elif isinstance(datos, str):
        # Convertir la cadena binaria individual a decimal
        return int(datos, 2)
    else:
        raise ValueError(
            f"El parámetro 'datos' debe ser una cadena o una lista de cadenas. y no {type(datos)}"
        )


import numpy as np


def get_espacio_matriz(espacio, normal, binario):
    """
    Funcion que retorna en una matriz el espacio de busqueda con los valores reales, los valores normalizados y los valores en binarios

    Parametros:
        espacio (np array): es el conjunto solucion original
        normal (numpy array): es el conjunto solucion normalizado
        binario (numpy array): es el conjunto solucion codificado a binario

    Retorna:
        espacio_matriz (numpy 2d array): retorna en una sola matriz los parametros, el orden de columnas es: espacio, normal, binario
    """

    espacio_matriz = np.column_stack((espacio, normal, binario))
    return espacio_matriz


def generar_poblacion(datos, ndatos):
    """
    Genera una población seleccionando aleatoriamente un subconjunto de elementos de los datos originales, devuelve tanto los datos seleccionados como sus índices correspondientes.

    Parámetros:
    - datos (iterable o numpy array): Conjunto de datos del cual se seleccionarán elementos.
    - ndatos (int): Número de elementos que se seleccionarán aleatoriamente para formar la población.

    Retorna:
    - numpy array: Arreglo de `ndatos` elementos seleccionados aleatoriamente del conjunto original `datos`.
    """

    n = len(datos)
    indices = np.random.choice(range(n), ndatos, replace=False)

    poblacion = np.zeros((ndatos, 2), dtype=object)

    poblacion[:, 0] = indices
    poblacion[:, 1] = [datos[i] for i in indices]
    # poblacion = np.column_stack((indices, [datos[i] for i in indices]))
    return poblacion
