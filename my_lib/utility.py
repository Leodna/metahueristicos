import numpy as np
import random


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


def ordenar_poblacion(poblacion, apt_column=3):
    pob_ord = poblacion.copy()

    n = len(pob_ord)
    for i in range(n):
        for j in range(0, n - i - 1):
            actual = pob_ord[j, apt_column]
            prox = pob_ord[j + 1, apt_column]

            if actual < prox:
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


"""OPERADORES DE SELECCIÓN"""


def torneo(T):
    k = len(T)
    ganador = T[0]
    for i in range(1, k):
        print(ganador)


# def seleccion_torneo(pob,k,n=1):


def ruleta(poblacion, n=2):
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
    indices = np.searchsorted(prob_acum, np.random.rand(n))
    return indiv_unicos[indices]


def seleccion_ruleta(pob_tam, poblacion, op_cruza=0):
    descendencia = []
    for i in range(pob_tam):
        padres = ruleta(poblacion)

        if op_cruza == 0:
            # Cruza un corte
            h1, h2 = cruce_un_corte(padres)
        elif op_cruza == 1:
            # cruza 2 cortes
            h1, h2 = cruce_dos_corte(padres)
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
        else:
            # Cruza homogenica
            h1, h2 = cruce_homogenea(padres)

        descendencia.append([h1, padres[:, 2]])
        descendencia.append([h2, padres[:, 2]])

    return descendencia


def seleccion_poligamica(poblacion, op_cruza=0):
    # número total de inidividuos
    n = len(poblacion)
    # número de cruces para pasar genes
    n_cruces = int(n / 2)

    for i in range(n_cruces):
        # selección de índices consecutivos para un par de padres y asegura que no se repita la selección
        p1 = random.randint(0, n)
        p2 = random.randint(0, n)
        padres = np.vstack([poblacion[p1], poblacion[p2]])

        if op_cruza == 0:
            # Cruza un corte
            h1, h2 = cruce_un_corte(padres)
        elif op_cruza == 1:
            # cruza 2 cortes
            h1, h2 = cruce_dos_corte(padres)
        else:
            # cruza homogenica
            h1, h2 = cruce_homogenea(padres)

        descendencia.append([h1, padres[:, 2]])
        descendencia.append([h2, padres[:, 2]])

    return descendencia


def seleccion_rank(poblacion, op_cruza=0):
    n = len(poblacion)
    descendencia = []
    for i in range(0, n, 2):
        padres = np.vstack([poblacion[i], poblacion[i + 1]])

        if op_cruza == 0:
            # Cruza un corte
            h1, h2 = cruce_un_corte(padres)
        elif op_cruza == 1:
            # cruza 2 cortes
            h1, h2 = cruce_dos_corte(padres)
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

    cromos_p1 = np.array([p1[: int(n_cromos / 2)], p1[-int(n_cromos / 2) :]])
    cromos_p2 = np.array([p2[: int(n_cromos / 2)], p2[-int(n_cromos / 2) :]])

    h1 = f"{cromos_p2[0]}{cromos_p1[1]}"
    h2 = f"{cromos_p1[0]}{cromos_p2[1]}"

    return h1, h2


def cruce_dos_corte(padres):
    n_cromos = len(padres[0, 2])
    p1 = padres[0, 2]
    p2 = padres[1, 2]

    # división de padre y madre en tre partes
    ter1 = int(n_cromos / 3)
    ter2 = ter1 * 2

    cromos_p = np.array([p1[:ter1], p1[ter1:ter2], p1[ter2:]])

    cromos_m = np.array([p2[:ter1], p2[ter1:ter2], p2[ter2:]])

    h1 = f"{cromos_m[0]}{cromos_p[1]}{cromos_m[2]}"
    h2 = f"{cromos_p[0]}{cromos_m[1]}{cromos_p[2]}"
    # print([hijo1,hijo2])
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


"""OBTENER SIGUIENTE GENERACION"""


def get_next_gen(poblacion, op_seleccion, op_cruza=0):
    if op_seleccion == 0:
        # Seleccion por ruelta
        # Numero de parajas a reproducir
        n = int(len(poblacion) / 2)
        return seleccion_ruleta(n, poblacion, op_cruza=op_cruza)
    elif op_seleccion == 1:
        # selección aleatoria monogamica
        return seleccion_monogamica(poblacion, op_cruza=op_cruza)
    elif op_seleccion == 2:
        # selección aleatoria poligamica
        return seleccion_poligamica(poblacion, op_cruza=op_cruza)
    elif op_seleccion == 3:
        # Selección por ranking
        return seleccion_rank(poblacion, op_cruza=op_cruza)
    else:
        # TO-DO implementar todo el metodo de selección por
        # Torneo
        return None


"""OPERADORES DE MUTACION"""


def mutacion(individuo):
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


def paro_epsilon(pob, threshold=0.75, majority_th=0.8):
    """
    Parametros:
    - pob: población de inidividuos
    - threshold: umbral de la función de costo (default: 0.75)
    - majority_th: proporción de la población que debe cumplir con el umbral (default: 0.8)
    Retorna:
    - True si se cumple la condición de paro, False en caso contrario
    """

    porporcion = np.mean(pob[:, 3] > threshold)
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
