from enum import Enum


class ModelEnum(Enum):
    DCGAN = 'DCGAN'
    CGAN = 'CGAN'
    DDPM = 'DDPM'
    DDIM = 'DDIM'
    BIGGAN = 'BIGGAN'

    def __str__(self):
        return self.value