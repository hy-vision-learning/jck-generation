from enum import Enum


class ModelEnum(Enum):
    DCGAN = 'DCGAN'
    CGAN = 'CGAN'

    def __str__(self):
        return self.value