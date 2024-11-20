from .slot import Slot
from .package import Package
from typing import List


class Container:
    def __init__(self):
        self.height = 12
        self.width = 2.4
        self.slot_width = 0.8
        self.slots: List[Slot] = self._create_slots()

    def _create_slots(self) -> List[Slot]:
        # Calcular la cantidad de slots que caben en el ancho del contenedor
        num_slots = int((self.width * 10) / (self.slot_width * 10))
        return [Slot(self.height, self.slot_width) for _ in range(num_slots)]

    def llenar_container(self, packages: List[Package]) -> List[Package]:
        """
        Llena el container con paquetes de una lista.
        :param packages: Lista de paquetes a almacenar
        :return: Lista de paquetes restantes que no pudieron ser almacenados
        """
        remaining_packages = packages.copy()  # Copia de la lista de paquetes
        for package in packages:
            stored = False
            # Intentar almacenar el paquete en cualquier slot disponible
            for slot in self.slots:
                if slot.store_package(package):
                    remaining_packages.remove(package)
                    stored = True
                    break
            if not stored:
                # Si el paquete no cabe en ningún slot, pasa al siguiente paquete
                continue

        return remaining_packages

    def get_packages_in_container(self) -> List[Package]:
        aux_package: List[Package] = []
        for slot in self.slots:
            aux_package += slot.get_packages()
        return aux_package

    def get_slots(self) -> List[Slot]:
        return self.slots

    def __str__(self):
        return f"Container(height={self.height}, width={self.width}, slots={len(self.slots)})"

    def __repr__(self):
        return self.__str__()
