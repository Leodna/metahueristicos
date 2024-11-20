from typing import List, Tuple
import random
from .individuo import Individuo
from .container import Container


class GA:
    def __init__(self, population: List[Individuo], cross_rate=0.7, mut_rate=0.1):
        """
        Inicializa el algoritmo genético.
        :param population: Lista inicial de individuos.
        :param cross_rate: Probabilidad de cruce (0.0 a 1.0).
        :param mut_rate: Probabilidad de mutación (0.0 a 1.0).
        """
        self.population = population
        self.cross_rate = cross_rate
        self.mut_rate = mut_rate

    def selection(self) -> Individuo:
        """
        Realiza la selección por ruleta basada en aptitud.
        :return: Individuo seleccionado.
        """
        aptitudes = [individuo.aptitud for individuo in self.population]
        total_aptitud = sum(aptitudes)
        if total_aptitud == 0:
            return random.choice(
                self.population
            )  # Selección aleatoria si todas las aptitudes son 0
        selection = random.uniform(0, total_aptitud)
        acumulado = 0
        for individuo in self.population:
            acumulado += individuo.aptitud
            if acumulado >= selection:
                return individuo

    def crossover(
        self, padre1: Individuo, padre2: Individuo
    ) -> Tuple[Individuo, Individuo]:
        """
        Realiza un cruce entre dos individuos utilizando intercambio de contenedores.
        Si el cromosoma tiene un solo contenedor, no realiza el cruce.
        :param padre1: Primer individuo.
        :param padre2: Segundo individuo.
        :return: Dos nuevos individuos hijos.
        """
        # Verificar que ambos individuos tienen más de un contenedor
        if len(padre1.get_cromosoma()) <= 1 or len(padre2.get_cromosoma()) <= 1:
            # Si no se puede realizar el cruce, devolver copias de los padres
            return padre1, padre2

        # Elegir un punto de cruce
        punto_cruce = random.randint(1, len(padre1.get_cromosoma()) - 1)

        # Intercambiar segmentos
        hijo1_cromosoma = (
            padre1.get_cromosoma()[:punto_cruce] + padre2.get_cromosoma()[punto_cruce:]
        )
        hijo2_cromosoma = (
            padre2.get_cromosoma()[:punto_cruce] + padre1.get_cromosoma()[punto_cruce:]
        )

        return Individuo(hijo1_cromosoma), Individuo(hijo2_cromosoma)

    def mutar(self, individuo: Individuo):
        """
        Realiza una mutación aleatoria en un individuo.
        :param individuo: Individuo a mutar.
        """
        if random.random() < self.mut_rate:
            cromosoma = individuo.get_cromosoma()
            if cromosoma:
                # Selecciona un contenedor y lo reordena aleatoriamente
                contenedor_mutado = random.choice(cromosoma)
                paquetes = contenedor_mutado.get_packages_in_container()
                random.shuffle(paquetes)
                nuevos_contenedores = self.fill_containers(paquetes)
                individuo.cromosoma = nuevos_contenedores
                individuo.aptitud = individuo.get_aptitud()  # Recalcular aptitud

    def evolucionar(self):
        """
        Realiza un ciclo completo de selección, cruce y mutación sobre la población.
        """
        nueva_population = []

        while len(nueva_population) < len(self.population):
            # Selección
            padre1 = self.selection()
            padre2 = self.selection()

            # Cruce
            if random.random() < self.cross_rate:
                hijo1, hijo2 = self.crossover(padre1, padre2)
            else:
                hijo1, hijo2 = padre1, padre2

            # Mutación
            self.mutar(hijo1)
            self.mutar(hijo2)

            # Agregar los hijos a la nueva población
            nueva_population.append(hijo1)
            if len(nueva_population) < len(self.population):  # Evitar exceder el tamaño
                nueva_population.append(hijo2)

        self.population = nueva_population

    def fill_containers(self, packages: List) -> List:
        """
        Utiliza el método de llenar contenedores similar al del Main.
        """
        containers = []
        remaining_packages = packages.copy()
        while remaining_packages:
            container = Container()
            remaining_packages = container.llenar_container(remaining_packages)
            containers.append(container)
        return containers

    def get_mejor_individuo(self) -> Individuo:
        """
        Devuelve el individuo con la mejor aptitud.
        """
        return max(self.population, key=lambda individuo: individuo.aptitud)

    def __str__(self):
        """
        Representación de la población actual.
        """
        return "\n".join([str(individuo) for individuo in self.population])
