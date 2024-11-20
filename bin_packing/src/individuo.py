from .container import Container
from typing import List


class Individuo:
    def __init__(self, lista_containers: List[Container]):
        self.cromosoma = lista_containers
        self.aptitud = self.__fitness_combined__()

    def __fitness_combined__(self, alpha=0.5, beta=0.5):
        """
        Función de aptitud combinada que minimiza el número de contenedores y maximiza el uso del espacio.
        :param alpha: Peso del objetivo de minimizar contenedores
        :param beta: Peso del objetivo de maximizar uso del espacio
        :return: Valor de aptitud
        """
        # Minimizar número de contenedores
        num_containers = len(self.cromosoma)
        fitness_containers = 1 / num_containers if num_containers > 0 else 0

        # Maximizar uso del espacio
        total_space = 0
        used_space = 0

        for container in self.cromosoma:
            for slot in container.get_slots():
                total_space += slot.total_height
                used_space += sum(package.height for package in slot.get_packages())

        fitness_space = used_space / total_space if total_space > 0 else 0

        # Combinación ponderada
        return alpha * fitness_containers + beta * fitness_space

    def get_cromosoma(self) -> List[Container]:
        return self.cromosoma

    def show_cromosomas(self):
        for i, alelo in enumerate(self.cromosoma):
            print(f"Contenedor {i+1}:")
            for j, slot in enumerate(alelo.get_slots()):
                print(f"    Slot {j+1}: {slot}")

    def get_aptitud(self):
        return self.aptitud

    def __str__(self):
        return f"Aptitud: {self.aptitud}"

    def __repr__(self):
        return self.__str__()
