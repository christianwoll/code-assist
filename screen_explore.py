import time
import random
import numpy as np
from PIL import Image
from mss import mss
from tile_encoder import LowerTileEncoder, UpperTileEncoder




sct = mss()

def grab_tile(x, y, width=64, height=64):
    sct_img = sct.grab({'top': y, 'left': x, 'width': width, 'height': height})

    print('<screenshot taken>')

    pixels = np.array(sct_img)
    grayscale = np.dot(pixels[...,:3], [0.299, 0.587, 0.114])

    return grayscale



tile_size = 64
x, y = 100, 100
tiles = [grab_tile(x, y)]*64

upperTileEncoder = UpperTileEncoder()
lowerTileEncoder = LowerTileEncoder()

while True:
    x = x + random.randrange(5) - 2
    y = y + random.randrange(5) - 2

    x = max(0, min(1216, x))
    y = max(0, min(736, y))

    print(str(len(tiles)) + ' tiles.')
    print('@ ' + str(x) + ', ' + str(y))

    tiles.append(grab_tile(x, y))
    tiles = tiles[-10000:]

    if len(tiles) > 1000:
        sample = random.sample(tiles, 1000)
        upperTileEncoder.fit(sample)
        upperTileEncoder.save()
        
        chips = upperTileEncoder.encode(sample)
        lowerTileEncoder.fit(chips)
        lowerTileEncoder.save()
