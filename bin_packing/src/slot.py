from .package import Package

from typing import List, Dict


class Slot:
    def __init__(self, height, width):
        self.total_height = height
        self.width = width
        self.available_height = height  # Altura disponible en el slot
        self.packages: List[Package] = []  # Paquetes almacenados en el slot
        self.inventory: Dict[int, int] = {
            2: 0,
            3: 0,
            5: 0,
            6: 0,
            8: 0,
        }  # Variable que indica cuantos paquetes de cada dimensión hay [2cm,3cm,5cm,6cm,8cm]

    def store_package(self, package) -> bool:
        """
        Intenta almacenar un paquete en el slot si hay espacio suficiente.
        :param package: Objeto de tipo Package
        :return: True si se almacena, False si no cabe
        """
        if package.height <= self.available_height and package.width <= self.width:
            self.packages.append(package)
            self.inventory[package.get_height()] += 1
            self.available_height -= package.height  # Reducir la altura disponible
            return True
        return False

    def get_packages(self) -> List[Package]:
        return self.packages

    def __str__(self):
        return f"Slot(total_height={self.total_height}, available_height={self.available_height}, packages={self.packages}, {len(self.packages)})"
        # return f"Slot(total_height={self.total_height}, available_height={self.available_height}, packages=[{self.inventory[8]}, {self.inventory[6]}, {self.inventory[5]}, {self.inventory[3]}, {self.inventory[2]}], {len(self.packages)})"

    def __repr__(self):
        return self.__str__()
