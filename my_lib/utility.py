import numpy as np
import random
from math import radians, sin, cos, sqrt, atan2
import itertools


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


def generar_poblacion_perm(datos, tam_pob, n_cromas=None):
    if not n_cromas:
        n_cromas = len(datos)

    n = n_cromas
    poblacion = np.zeros((tam_pob, n))
    for i in range(tam_pob):
        poblacion[i] = np.random.permutation(n)

    return poblacion


def individuo_toString(inviduo):
    aux = np.array2string(inviduo, separator="")
    aux = aux.strip("[]")
    return aux


def ordenar_poblacion(poblacion, apt_column=3, reverse=False):
    pob_ord = poblacion.copy()

    n = len(pob_ord)
    for i in range(n):
        for j in range(0, n - i - 1):
            actual = pob_ord[j, apt_column]
            prox = pob_ord[j + 1, apt_column]

            if (actual < prox and not reverse) or (actual > prox and reverse):
                aux = pob_ord[j].copy()
                pob_ord[j] = pob_ord[j + 1].copy()
                pob_ord[j + 1] = aux.copy()

    return pob_ord


# def cruzar_individuos(pob, espacio, p=0, m=1):
#     n = len(pob)
#     n_cromos = len(pob[0, 2])

#     nueva_pob = pob.copy()

#     for i in range(0, n, 2):
#         padre = pob[i, 2]
#         madre = pob[i + 1, 2]

#         cromos_p = np.array([padre[: int(n_cromos / 2)], padre[-int(n_cromos / 2) :]])
#         cromos_m = np.array([madre[: int(n_cromos / 2)], madre[-int(n_cromos / 2) :]])

#         aux_1 = f"{cromos_p[p]}{cromos_m[m]}"
#         aux_2 = f"{cromos_m[m]}{cromos_p[p]}"

#         hijo_1 = buscar_individuo(espacio, aux_1)
#         hijo_2 = buscar_individuo(espacio, aux_2)
#         nueva_pob = np.vstack([nueva_pob, hijo_1])
#         nueva_pob = np.vstack([nueva_pob, hijo_2])

#     return nueva_pob


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


def hamming_dist(ind1, ind2):
    """
    Aplica la distancia de hamming entre dos individuos

    Parámetros
    ind1 (numpy array): La lista de genes de el individuo 1
    ind2 (numpy array): La lista de genes de el individuo 2

    Descripción
    Realiza una comparación entre los elementos de cada individuos, si existen elementos iguales la distancia aumenta en uno, si no, la distancia no aumenta. La distancia minima posible es 0, lo cual indicaría que ambos individuos no comparten genes.

    Retoran
    distancia (numerico): La distancia de hamming
    """
    distancia = 0
    for i in range(len(ind1)):
        if ind1[i] != ind2[i]:
            distancia += 1
    return distancia


def diversity_rate(poblacion):
    aux_pob = poblacion[:, 0]
    N = len(aux_pob)
    D = 0
    n_comparaciones = 0

    # Sumar la distancia de Hamming entre todos los pares de individuos
    for i in range(N):
        for j in range(i + 1, N):
            D += hamming_dist(aux_pob[i], aux_pob[j])
            n_comparaciones += 1

    # Calcular la distancia promedio
    if n_comparaciones != 0:
        D_promedio = D / n_comparaciones
    else:
        D_promedio = 0

    return D_promedio


"""" GET DISTANCIA """


def get_distancia(lat1, long1, lat2, long2):
    # radio de la tierra (km)
    r = 6371.0
    # conversión de grados a radianes
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    # diferencias
    dlat = lat1 - lat2
    dlong = long1 - long2

    # formula de la distancia de HAversine
    # calcula la distancia entre dos puntos en una esfera, la tierra
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlong / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distancia = r * c
    return distancia


"""OPERADORES DE SELECCIÓN"""


def torneo(T):
    k = len(T)
    ganador = T[0]
    for i in range(1, k):
        print(ganador)


# def seleccion_torneo(pob,k,n=1):


def ruleta(poblacion, n=2, mode="max"):
    # Valores unicos
    _, indices_unicos = np.unique(poblacion[:, 1], return_index=True)
    indiv_unicos = poblacion[indices_unicos]

    if mode == "max":
        # Ordenar por aptitud
        indiv_unicos = ordenar_poblacion(poblacion)
        # Aptitud total
        apt_total = np.sum(indiv_unicos[:, 3])
        # Probabilidad por aptitud
        prob_apt = indiv_unicos[:, 3] / apt_total
    else:
        # Ordenar por aptitud
        indiv_unicos = ordenar_poblacion(poblacion, reverse=True)
        # Aptitud total
        apt_total = np.sum(1 / indiv_unicos[:, 3])
        # Probabilidad por aptitud
        prob_apt = (1 / indiv_unicos[:, 3]) / apt_total

    # Probabilidad acumulada
    prob_acum = np.cumsum(prob_apt)
    indices = np.searchsorted(prob_acum, np.random.rand(n))
    return indiv_unicos[indices]


def seleccion_ruleta(pob_tam, poblacion, op_cruza=0, mode="max"):
    descendencia = []
    for i in range(pob_tam):
        padres = ruleta(poblacion, mode=mode)

        if op_cruza == 0:
            # Cruza un corte
            h1, h2 = cruce_un_corte(padres)
        elif op_cruza == 1:
            # cruza 2 cortes
            h1, h2 = cruce_dos_corte(padres)
        elif op_cruza == 2:
            # Cruza homogenica
            h1, h2 = cruce_homogenea(padres)
        elif op_cruza == 3:
            # Cruza pmx
            h1, h2 = cruce_pmx(padres)
        elif op_cruza == 4:
            # Cruza cx
            h1, h2 = cruce_cx(padres)
        else:
            # cruza homogenica
            h1, h2 = cruce_homogenea(padres)

        descendencia.append([h1, padres[:, 2]])
        descendencia.append([h2, padres[:, 2]])

    return descendencia


def seleccion_monogamica(poblacion, op_cruza=0):
    # número total de individuos de la poblacion
    n = len(poblacion)
    # ordenar de forma aleatoria la poblacion
    pob = poblacion.copy()
    np.random.shuffle(pob)
    # lista para almacenar los hijos de cada padre
    descendencia = []
    for i in range(0, n, 2):
        padres = np.vstack([pob[i], pob[i + 1]])

        if op_cruza == 0:
            # Cruza un corte
            h1, h2 = cruce_un_corte(padres)
        elif op_cruza == 1:
            # Cruza 2 cortes
            h1, h2 = cruce_dos_corte(padres)
        elif op_cruza == 2:
            # Cruza homegincca
            h1, h2 = cruce_homogenea(padres)
        elif op_cruza == 3:
            # Cruza pmx
            h1, h2 = cruce_pmx(padres)
        elif op_cruza == 4:
            # Cruza cx
            h1, h2 = cruce_cx(padres)
        else:
            # Cruza homogenica
            h1, h2 = cruce_homogenea(padres)

        descendencia.append([h1, padres[:, 2]])
        descendencia.append([h2, padres[:, 2]])

    return descendencia


def seleccion_poligamica(poblacion, op_cruza=0):
    # número total de inidividuos
    n = len(poblacion)
    # número de parejas a formar
    n_parejas = int(n / 2)
    descendencia = []
    for i in range(n_parejas):
        # selección de índices consecutivos para un par de padres y asegura que no se repita la selección
        p1 = np.random.randint(0, n)
        p2 = np.random.randint(0, n)
        padres = np.vstack([poblacion[p1], poblacion[p2]])

        if op_cruza == 0:
            # Cruza un corte
            h1, h2 = cruce_un_corte(padres)
        elif op_cruza == 1:
            # cruza 2 cortes
            h1, h2 = cruce_dos_corte(padres)
        elif op_cruza == 2:
            # cruza homogenica
            h1, h2 = cruce_homogenea(padres)
        elif op_cruza == 3:
            # cruza pmx
            h1, h2 = cruce_pmx(padres)
        elif op_cruza == 4:
            # cruza cx
            h1, h2 = cruce_cx(padres)
        else:
            # cruza homogenica
            h1, h2 = cruce_homogenea(padres)

        descendencia.append([h1, padres[:, 2]])
        descendencia.append([h2, padres[:, 2]])

    return descendencia


def seleccion_rank(poblacion, op_cruza=0, mode="max"):
    n = len(poblacion)
    descendencia = []

    if mode == "max":
        poblacion = ordenar_poblacion(poblacion)
    else:
        poblacion = ordenar_poblacion(poblacion, reverse=True)

    for i in range(0, n, 2):
        padres = np.vstack([poblacion[i], poblacion[i + 1]])

        if op_cruza == 0:
            # Cruza un corte
            h1, h2 = cruce_un_corte(padres)
        elif op_cruza == 1:
            # cruza 2 cortes
            h1, h2 = cruce_dos_corte(padres)
        elif op_cruza == 2:
            # cruza homogenica
            h1, h2 = cruce_homogenea(padres)
        elif op_cruza == 3:
            # cruza  pmx
            h1, h2 = cruce_pmx(padres)
        elif op_cruza == 4:
            h1, h2 = cruce_cx(padres)
        else:
            # cruza homogenica
            h1, h2 = cruce_homogenea(padres)

        descendencia.append([h1, padres[:, 2]])
        descendencia.append([h2, padres[:, 2]])

    return descendencia


"""OPERADORES DE CRUCE"""


def cruce_un_corte(padres):
    n_cromos = len(padres[0, 2])
    p1 = padres[0, 2]
    p2 = padres[1, 2]

    # cromos_p1 = np.array([p1[: int(n_cromos / 2)], p1[-int(n_cromos / 2) :]])
    # cromos_p2 = np.array([p2[: int(n_cromos / 2)], p2[-int(n_cromos / 2) :]])
    cromos_p1 = np.array([p1[: int(n_cromos / 2)], p1[int(n_cromos / 2) :]])
    cromos_p2 = np.array([p2[: int(n_cromos / 2)], p2[int(n_cromos / 2) :]])

    h1 = f"{cromos_p2[0]}{cromos_p1[1]}"
    h2 = f"{cromos_p1[0]}{cromos_p2[1]}"

    return h1, h2


def cruce_dos_corte(padres):
    # esta función ya realiza la separación adecuada para las tercias sin importar si
    # es par o no la longitud de los cromosoams
    n_cromos = len(padres[0, 2])
    p1 = padres[0, 2]
    p2 = padres[1, 2]

    # Convertimos las cadenas de cromosomas en listas de genes
    p1 = list(map(str, p1))  # Convertir a lista de caracteres
    p2 = list(map(str, p2))

    # calcula la longitud de cada parte
    part_length = n_cromos // 3
    remaining = n_cromos % 3

    # divide los cromosomas en tres partes
    parts_p = []
    parts_m = []
    start = 0
    for i in range(3):
        end = start + part_length
        if i < remaining:
            end += 1
        parts_p.append(p1[start:end])
        parts_m.append(p2[start:end])
        start = end

    h1 = f"{parts_m[0]}{parts_p[1]}{parts_m[2]}"
    h2 = f"{parts_p[0]}{parts_m[1]}{parts_p[2]}"

    return h1, h2


def cruce_homogenea(padres):
    n_cromos = len(padres[0, 2])
    p1 = padres[0, 2]
    p2 = padres[1, 2]

    h1 = []
    h2 = []

    for i in range(n_cromos):
        # selecciona de forma aleatoria de que padre/madre proviene el bit para cada hijo
        if random.random() < 0.5:
            h1.append(p1[i])
            h2.append(p2[i])
        else:
            h1.append(p2[i])
            h2.append(p1[i])

    # Convertir las listas de bits a cadenas
    h1 = "".join(h1)
    h2 = "".join(h2)
    return h1, h2


def cruce_pmx(padres, target_col=0):
    """
    Realiza una cruza PMX (cruza parcialmente mapeada) en dos cromosomas padres

    Args:
    padres (np.arraylist): Lista con los dos padres a cruzar
    target_col(int): Indice de la columna en donde se encuentra el individiduo codificado, por defecto se toma la columna 0

    Returns:
    tupla (np.arraylist): Los dos hijos producto de PMX, en forma de np list
    """
    p1 = padres[0, 0]
    p2 = padres[1, 0]

    if len(p1) != len(p2):
        raise Exception("Los padres no tienen la misma cantidad de cromosomas")

    p1 = np.array(list(p1)).astype(int)
    p2 = np.array(list(p2)).astype(int)

    n_cromos = len(p2)

    # Seleccionar puntos de corte aleatorios
    xpt1, xpt2 = np.random.randint(1, n_cromos - 1, 2)

    while xpt1 == xpt2:
        xpt2 = np.random.randint(1, n_cromos - 1)
        # print(xpt2)

    if xpt1 > xpt2:
        xpt1, xpt2 = xpt2, xpt1
    # print(f"Padre1: {p1}")
    # print(f"Padre2: {p2}")
    # print(f"pt1 : {xpt1}")
    # print(f"pt2 : {xpt2}")

    # Separar los cromosomas de los padres, por los pts de corte generados
    cromos_p1 = [p1[:xpt1], p1[xpt1:xpt2], p1[xpt2:]]
    cromos_p2 = [p2[:xpt1], p2[xpt1:xpt2], p2[xpt2:]]

    # Matriz variabilidad
    offspring = [cromos_p2[1], cromos_p1[1]]

    # proto-hijos
    proto_child1 = [cromos_p1[0], cromos_p2[1], cromos_p1[2]]
    proto_child2 = [cromos_p2[0], cromos_p1[1], cromos_p2[2]]

    # print(f"proto-hijo1: {proto_child1}")
    # print(f"proto-hijo2: {proto_child2}")

    for i, (child) in enumerate(zip([proto_child1, proto_child2])):
        pchild = child[0]
        for j in range(0, 3, 2):
            segment = pchild[j]
            for k in range(len(segment)):
                cromosoma = segment[k]
                while cromosoma in offspring[i]:
                    cid = np.where(offspring[i] == cromosoma)[0]
                    if cid.size > 0:
                        cid = cid[0]
                        # print(
                        #     f"El elemento '{cromosoma}' está en la posición: {cid} cambiando por {cromosoma} -> {offspring[i-1][cid]}"
                        # )
                        cromosoma = offspring[i - 1][cid]

                segment[k] = cromosoma

            pchild[j] = segment

    # print(f"final-hijo1: {proto_child1}")
    # print(f"final-hijo2: {proto_child2}")

    h1 = np.concatenate(proto_child1)
    h2 = np.concatenate(proto_child2)
    return h1, h2


def cruce_cx(padres):
    """
    Realiza un cruce CX (Cyclic Crossover) en la población con numpy arrays.

    Args:
    padres: Un arreglo numpy de dos dimensiones que contiene dos individuos (padres).
            Cada padre es un vector de cromosomas (genes).

    Returns:
    offspring1: Primer hijo resultado del cruce.
    offspring2: Segundo hijo resultado del cruce.

    """
    p1 = padres[0, 0]
    p2 = padres[1, 0]

    if len(p1) != len(p2):
        raise Exception("Los padres no tienen la misma cantidad de cromosomas")

    p1 = np.array(list(p1)).astype(int)
    p2 = np.array(list(p2)).astype(int)

    print(f"Padre 1: {p1}")
    print(f"Padre 2: {p2}")

    # n_cromos = len(p1)

    size = len(p1)
    offspring1 = np.full(size, -1)
    offspring2 = np.full(size, -1)

    # diccionario para los indices ciclados
    p2_indices = {val: idx for idx, val in enumerate(p2)}

    # posiciones para el cx, permutaciones
    start = 0
    while -1 in offspring1:
        if offspring1[start] == -1:
            cycle_indices = []
            current = start

            # inicio del ciclo
            while current not in cycle_indices:
                cycle_indices.append(current)
                current_value = p1[current]
                # índice para el padre 2
                current = p2_indices[current_value]

            # intercambio de los valores según el ciclo que se haya encontrado
            for index in cycle_indices:
                offspring1[index] = p1[index]
                offspring2[index] = p2[index]

            # cambia el ciclo de los restantes
            start += 1

    # se llena el resto d elos valores que no están en el ciclo
    for i in range(size):
        if offspring1[i] == -1:
            offspring1[i] = p2[i]
        if offspring2[i] == -1:
            offspring2[i] = p1[i]

    print(f"Hijo 1: {offspring1}")
    print(f"Hijo 2: {offspring2}")
    return offspring1, offspring2


"""OBTENER SIGUIENTE GENERACION"""


def get_next_gen(poblacion, op_seleccion, op_cruza=0, mode="max"):
    if op_seleccion == 0:
        # Seleccion por ruelta
        # Numero de parajas a reproducir
        n = int(len(poblacion) / 2)
        return seleccion_ruleta(n, poblacion, op_cruza=op_cruza, mode=mode)
    elif op_seleccion == 1:
        # selección aleatoria monogamica
        return seleccion_monogamica(poblacion, op_cruza=op_cruza)
    elif op_seleccion == 2:
        # selección aleatoria poligamica
        return seleccion_poligamica(poblacion, op_cruza=op_cruza)
    elif op_seleccion == 3:
        # Selección por ranking
        return seleccion_rank(poblacion, op_cruza=op_cruza, mode=mode)
    else:
        # TO-DO implementar todo el metodo de selección por
        # Torneo
        return None


"""OPERADORES DE MUTACION"""


def mutation(
    pob,
    indiv_percent=10,
    cromosomas_mutation=None,
    opt=0,
    operator=0,
    funcion_aptitud=None,
    reverse=True,
):
    """
    Aplica una mutación a una parte de la población original.

    Parámetros:
    pob (array): La población original que se va a mutar.
    indiv_percent (int): El porcentaje de la población que se va a mutar. Debe ser un valor entre 0 y 10. (default: 10)
    cromosomas_mutation (int): La cantidad de genes que se van a mutar en cada individuo. Si no se proporciona, se tomará la mitad de los cromosomas del individuo. (default: None)
    opt (int): La opción para seleccionar los individuos a mutar. Puede ser 0 (aleatoriedad) o 1 (por aptitud). (default: 0)
    operator (int): El operador de mutación a utilizar. (default: 0)

    Descripción:
    La función mutation aplica una mutación a una parte de la población original. Primero, se seleccionan los individuos a mutar según el porcentaje proporcionado y la opción seleccionada. Luego, se aplica la mutación a cada individuo seleccionado utilizando el operador de mutación especificado.

    Validación de errores:
    La función verifica que el porcentaje de la población a mutar sea mayor a 0 y menor o igual a 10. También verifica que la cantidad de genes a mutar no supere la mitad de los cromosomas del individuo.

    Retorno:
    La población resultante después de la mutación.

    Notas:
    * La función mutation modifica la población original, por lo que se recomienda trabajar con una copia de la población original si se desea preservar la población original.
    """
    # Validación de errores
    tam_pob = pob.shape[0]
    num_cromos = pob[0][0].shape[0]
    indiv_percent = int(indiv_percent)

    if indiv_percent <= 0 or indiv_percent > 10:
        raise ValueError(
            "El porcentaje de la población a mutar debe ser mayor a 0 y menor o igual a 10."
        )

    if not cromosomas_mutation:
        cromosomas_mutation = num_cromos // 2
    else:
        cromosomas_mutation = int(cromosomas_mutation)
        if cromosomas_mutation > (num_cromos // 2):
            raise ValueError(
                "La cantidad de genes a mutar no debe pasar la mitad de los cromosomas del individuo"
            )

    # Seleccionar individuos a mutar
    indiv_percent /= 100
    n = round(indiv_percent * tam_pob)
    # opt=0 ----> ALEATORIEDAD
    if opt == 0:
        indices_mutation = np.random.choice(tam_pob, n, replace=False)

    # opt=1 ----> Por aptitud
    else:
        pob = ordenar_poblacion(pob, reverse=not reverse)
        indices_mutation = np.arange(n)

    mutate = [heuristic_mutation, mutacion_scramble]

    for i, indice in enumerate(indices_mutation):
        individuo = pob[indice]
        mutante = mutate[operator](individuo[0], cromosomas_mutation, funcion_aptitud)
        # print(
        #     f"individuo original: {individuo[2]} (apt) {individuo[3]} vs mutante {mutante[2]} (apt) {mutante[3]}"
        # )
        pob[indice] = mutante

    return pob


def heuristic_mutation(
    individuo, cromosomas_mutation=None, funcion_aptitud=None, reverse=True
):
    """
    Aplica una mutación heurística a un individuo.

    Parámetros:
    individuo (array): El individuo que se va a mutar.
    cromosomas_mutation (int): La cantidad de genes que se van a mutar. Si no se proporciona, se tomará la mitad de los cromosomas del individuo. (default: None)

    Descripción:
    La función heuristic_mutation aplica una mutación heurística a un individuo. Primero, se selecciona un substring del individuo para mutar. Luego, se generan todas las posibles permutaciones del substring y se evalúa la aptitud de cada permutación. Finalmente, se selecciona la permutación con la mayor aptitud y se devuelve como el nuevo individuo mutado.

    Validación de errores:
    La función verifica que la cantidad de genes a mutar no supere la mitad de los cromosomas del individuo.

    Retorno:
    El individuo mutado con la aptitud optima.
    """
    num_cromos = len(individuo)
    # print("\n")
    # print(individuo)

    if not cromosomas_mutation:
        cromosomas_mutation = num_cromos // 2
    else:
        cromosomas_mutation = int(cromosomas_mutation)
        if cromosomas_mutation > (num_cromos // 2):
            raise ValueError(
                "La cantidad de genes a mutar no debe pasar la mitad de los cromosomas del individuo"
            )
    # obtener substring
    n = (num_cromos + 1) - cromosomas_mutation
    idx = np.random.randint(0, n)
    sublist = individuo[idx : idx + cromosomas_mutation]
    # print(sublist)
    # print("\n")

    # Generar posibles soluciones
    permutations = np.array(list(itertools.permutations(sublist)))

    permutation_pob = np.zeros((permutations.shape[0], 4), dtype="object")
    # permutation_pob[:,0] = [c.astype(int) for c in permutations]

    for i, p in enumerate(permutations):
        aux = individuo.copy()
        aux[idx : idx + cromosomas_mutation] = p
        permutation_pob[i, 0] = aux

    permutation_apt = funcion_aptitud(permutation_pob)
    permutation_pob[:, 3] = permutation_apt

    permutation_pob = ordenar_poblacion(permutation_pob, reverse=reverse)
    permutation_pob[0, 2] = " ".join([chr(c + 64) for c in permutation_pob[0, 0]])
    # print(permutation_pob)

    return permutation_pob[0], idx


def mutacion(individuo, cromosomas_mutation=None):
    """
    Mutación que cambia un bit aleatorio de cada individuo,
    cambiando el 0 por 1.
    """
    # convertir el individuo binario a una lista de bits
    bits = list(individuo)
    # seleccionar un índice aleatorio para mutar
    index = random.randint(0, 5)

    # mutar el bit en el índice seleccionado
    if bits[index] == "0":
        bits[index] = "1"
    else:
        bits[index] = "0"

    # convertir la lista de bits de vuelta a una cadena binaria
    individuo_mutado = "".join(bits)

    return individuo_mutado


def mutacion_scramble(individuo, cromosomas_mutation=None, funcion_aptitud=None):
    n_cromos = len(individuo)

    # selección de dos indices aleatorios
    idx1, idx2 = np.sort(np.random.choice(range(n_cromos), size=2, replace=False))
    # segmento a mutar
    scramble_part = individuo[idx1 : idx2 + 1]
    print(f"subrista:{scramble_part}")

    # desorden del segmente
    np.random.shuffle(scramble_part)
    print(f"subrista shuffle:{scramble_part}")

    # reemplazo de la parte del individuo con el segmento desordenado
    individuo[idx1 : idx2 + 1] = scramble_part

    return individuo, idx1


"""CRITERIOS DE PARO"""


def paro_epsilon_artesanal(pob, umb=0.8, prctg=0.8):
    tam_pob = len(pob)  # 100%
    paro = tam_pob * prctg  # Porcentaje de pob para parar
    count = 0  # Num de ind que son aptos

    for ind in pob:
        if ind[3] >= umb:  # Determinar si un ind es apto
            count += 1

    if count >= paro:  # Paro si hay individuos aptos
        return True

    return False


def paro_epsilon(pob, threshold=0.75, majority_th=0.8, mode="max"):
    """
    Parametros:
    - pob: población de inidividuos
    - threshold: umbral de la función de costo (default: 0.75)
    - majority_th: proporción de la población que debe cumplir con el umbral (default: 0.8)
    - opt: indica si es un problema de minimización o si es un problema de maximiazación, por defecto es maximizacion (default:0)
    Retorna:
    - True si se cumple la condición de paro, False en caso contrario
    """
    if opt == "max":
        porporcion = np.mean(pob[:, 3] > threshold)
    else:
        porporcion = np.mean(pob[:, 3] < threshold)

    if porporcion >= majority_th:
        return True

    return False


def paro_delta(pob, apt_prev, delta=1e-6):
    """
    Parametros:
    - pob: población de individuos
    - apt_prev: aptitud media de la población anterior
    - delta: valor umbral para considerar la convergencia (default: 1e-6)

    Retorna:
    - la mejor solución encontrada
    """
    apt_mean = np.mean(pob[:, 3])

    if abs(apt_mean - apt_prev) <= delta:
        return True

    return False
