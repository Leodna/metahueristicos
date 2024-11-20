class Package:
    VALID_HEIGHTS = {2, 3, 5, 6, 8}  # Alturas permitidas

    def __init__(self, height):
        if height not in Package.VALID_HEIGHTS:
            raise ValueError(
                f"El valor de height debe ser uno de {Package.VALID_HEIGHTS}"
            )
        self.height = height
        self.width = 0.8

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def __str__(self):
        return f"caja {self.height}"

    def __repr__(self):
        return f"caja {self.height}"
