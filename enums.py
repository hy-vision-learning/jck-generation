from enum import Enum


class ModelEnum(Enum):
    DCGAN = 'DCGAN'
    CGAN = 'CGAN'
    DDPM = 'DDPM'
    DDIM = 'DDIM'

    def __str__(self):
        return self.value