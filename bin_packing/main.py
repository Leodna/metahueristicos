from src import Package, Container, Slot, Individuo, GA
from typing import List

import random


class Main:

    def __init__(self, num_box1, num_box2, num_box3, num_box4, num_box5, tam_pob=10):
        self.num_box1 = num_box1
        self.num_box2 = num_box2
        self.num_box3 = num_box3
        self.num_box4 = num_box4
        self.num_box5 = num_box5
        self.tam_pob = tam_pob

    def run(self):
        lista_packages = self.sort_packages_by_height(self.get_package_list())
        population: List[Individuo] = []
        for i in range(self.tam_pob):
            aux_packages = lista_packages.copy()
            random.shuffle(aux_packages)
            population.append(Individuo(self.fill_containers(aux_packages)))

            # print(f"Individuo {i+1} {population[-1]}")
            # population[-1].show_cromosomas()
            # print("")
        genetico = GA(population=population, cross_rate=0.8, mut_rate=0.1)
        # Ejecutar generaciones
        num_generaciones = 50
        for i in range(num_generaciones):
            genetico.evolucionar()
            mejor_individuo = genetico.get_mejor_individuo()
            print(f"Generación {i+1}: Mejor aptitud = {mejor_individuo.aptitud:.4f}")

        # # Mejor individuo final
        # mejor_individuo_final = genetico.get_mejor_individuo()
        # print("Mejor Individuo Final:", mejor_individuo_final)
        # mejor_individuo_final.show_cromosomas()

    def get_package_list(self) -> List[Package]:
        """
        Genera una lista de paquetes con las cantidades especificadas por altura.

        :param num_height_2: Cantidad de paquetes de altura 2
        :param num_height_3: Cantidad de paquetes de altura 3
        :param num_height_5: Cantidad de paquetes de altura 5
        :param num_height_6: Cantidad de paquetes de altura 6
        :param num_height_8: Cantidad de paquetes de altura 8
        :return: Lista de paquetes generados
        """
        packages = []
        packages.extend([Package(2) for _ in range(self.num_box1)])
        packages.extend([Package(3) for _ in range(self.num_box2)])
        packages.extend([Package(5) for _ in range(self.num_box3)])
        packages.extend([Package(6) for _ in range(self.num_box4)])
        packages.extend([Package(8) for _ in range(self.num_box5)])
        return packages

    def sort_packages_by_height(self, packages: List[Package]) -> List[Package]:
        """
        Ordena una lista de paquetes de mayor a menor según su atributo height.

        :param packages: Lista de objetos de tipo Package
        :return: Lista de paquetes ordenada
        """
        return sorted(packages, key=lambda package: package.height, reverse=True)

    def fill_containers(self, packages: List[Package]) -> List[Container]:
        """
        Llena contenedores con los paquetes de la lista hasta que no quede ningún paquete.
        Crea nuevos contenedores según sea necesario.

        :param packages: Lista de paquetes a almacenar
        :return: Lista de contenedores llenados con los paquetes
        """
        containers: List[Container] = []  # Lista de contenedores llenados
        remaining_packages = (
            packages.copy()
        )  # Copiar la lista de paquetes para no modificarla directamente

        while remaining_packages:
            # Crear un nuevo contenedor
            container = Container()
            # Llenar el contenedor con los paquetes restantes
            remaining_packages = container.llenar_container(remaining_packages)
            # Agregar el contenedor a la lista
            containers.append(container)

        return containers


if __name__ == "__main__":
    main = Main(num_box1=20, num_box2=15, num_box3=10, num_box4=25, num_box5=28)
    main.run()
