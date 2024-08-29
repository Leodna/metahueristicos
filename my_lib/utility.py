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
    datos = np.rint(datos).astype(np.int64)
    # datos = datos.astype(np.int64)
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
    Genera una población seleccionando aleatoriamente un subconjunto de elementos de los datos originales, devuelve el subconjunto.

    Parámetros:
    - datos (iterable o numpy array): Conjunto de datos del cual se seleccionarán elementos.
    - ndatos (int): Número de elementos que se seleccionarán aleatoriamente para formar la población.

    Retorna:
    - numpy array: Arreglo de `ndatos` elementos seleccionados aleatoriamente del conjunto original `datos`.
    """

    n = len(datos)
    indices = np.random.choice(range(n), ndatos, replace=False)

    poblacion = np.zeros((ndatos, datos.shape[1]), dtype=object)

    poblacion[:, 0] = [datos[i, 0] for i in indices]
    poblacion[:, 1] = [datos[i, 1] for i in indices]
    poblacion[:, 2] = [datos[i, 2] for i in indices]
    poblacion[:, 3] = [datos[i, 3] for i in indices]

    return poblacion


def ordenar_poblacion(poblacion):
    pob_ord = poblacion.copy()

    n = len(pob_ord)
    for i in range(n):
        for j in range(0, n - i - 1):
            actual = pob_ord[j, 3]
            prox = pob_ord[j + 1, 3]

            if actual < prox:
                aux = pob_ord[j].copy()
                pob_ord[j] = pob_ord[j + 1].copy()
                pob_ord[j + 1] = aux.copy()

    return pob_ord


def cruzar_individuos(pob, espacio, p=0, m=1):
    n = len(pob)
    n_cromos = len(pob[0, 2])

    nueva_pob = pob.copy()

    for i in range(0, n, 2):
        padre = pob[i, 2]
        madre = pob[i + 1, 2]

        cromos_p = np.array([padre[: int(n_cromos / 2)], padre[-int(n_cromos / 2) :]])
        cromos_m = np.array([madre[: int(n_cromos / 2)], madre[-int(n_cromos / 2) :]])

        aux_1 = f"{cromos_p[p]}{cromos_m[m]}"
        aux_2 = f"{cromos_m[m]}{cromos_p[p]}"

        hijo_1 = buscar_individuo(espacio, aux_1)
        hijo_2 = buscar_individuo(espacio, aux_2)
        nueva_pob = np.vstack([nueva_pob, hijo_1])
        nueva_pob = np.vstack([nueva_pob, hijo_2])

    return nueva_pob


def buscar_individuo(espacio, individuo, columna=2):
    n = len(espacio)
    for i in range(n):
        if individuo == espacio[i, columna]:
            return espacio[i]
    print(individuo)
    return None


def buscar_hijos(hijos, espacio, columna=2):
    hijos_esp = []
    hijos_ind = [i[0] for i in hijos]
    hijos_ind = np.array(hijos_ind)

    for i, hijo in enumerate(hijos_ind):
        indice = np.where(espacio[:, columna] == hijo)
        indice = indice[0][0]
        hijos_esp.append(espacio[indice, :])
    return np.array(hijos_esp)


def seleccion_ruleta(poblacion):
    # Valores unicos
    _, indices_unicos = np.unique(poblacion[:, 1], return_index=True)
    indiv_unicos = poblacion[indices_unicos]

    # Ordenar por aptitud
    indiv_unicos = ordenar_poblacion(poblacion)

    # Aptitud total
    apt_total = np.sum(indiv_unicos[:, 3])
    # Probabilidad por aptitud
    prob_apt = indiv_unicos[:, 3] / apt_total
    # Probabilidad acumulada
    prob_acum = np.cumsum(prob_apt)

    indices = np.searchsorted(prob_acum, np.random.rand(2))
    return indiv_unicos[indices]


def cruce_un_corte(padres):
    n_genes = len(padres[0, 2])
    p1 = padres[0, 2]
    p2 = padres[1, 2]

    cromos_p1 = np.array([p1[: int(n_genes / 2)], p1[-int(n_genes / 2) :]])
    cromos_p2 = np.array([p2[: int(n_genes / 2)], p2[-int(n_genes / 2) :]])

    h1 = f"{cromos_p2[0]}{cromos_p1[1]}"
    h2 = f"{cromos_p1[0]}{cromos_p2[1]}"

    return h1, h2


def seleccionar_padres(pob_tam, poblacion):
    descendencia = []
    for i in range(pob_tam):
        padres = seleccion_ruleta(poblacion)
        cruce_un_corte(padres)
        h1, h2 = cruce_un_corte(padres)

        descendencia.append([h1, padres[:, 2]])
        descendencia.append([h2, padres[:, 2]])

    return descendencia
