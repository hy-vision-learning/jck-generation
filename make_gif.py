import os
from PIL import Image

path = './save/biggan/fqbiggan-18.7/sample'

images = [Image.open(os.path.join(path, 'image', str(i), f'fixed_samples{i}.jpg')) for i in range(1000, 56000, 1000)]
images[0].save(os.path.join(path, 'samples.gif'), save_all=True, append_images=images[1:], duration=250, loop=0)