from enum import Enum


class ModelEnum(Enum):
    DCGAN = 'DCGAN'
    CGAN = 'CGAN'
    DDIM = 'DDIM'

    def __str__(self):
        return self.value