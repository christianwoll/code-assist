import time
import random
import numpy as np
from PIL import Image
from mss import mss
import pytesseract
from sklearn.cluster import KMeans
from tile_encoder import LowerTileEncoder, UpperTileEncoder




sct = mss()

# BUG:
# The dict is modified in place by mss so make a copy for each call.
mon = {'top': 0, 'left': 0, 'width': 1280, 'height': 800}

def grab_tiles(num_tiles, tile_size=(64,64)):
    sct_img = sct.grab(dict(mon))

    print('<screenshot taken>')

    pixels = np.array(sct_img)
    grayscale = np.dot(pixels[...,:3], [0.299, 0.587, 0.114])

    tiles = []
    for _ in range(num_tiles):
        i = random.randrange(grayscale.shape[0] - tile_size[0])
        j = random.randrange(grayscale.shape[1] - tile_size[1])
        tile = grayscale[i:i+tile_size[0],j:j+tile_size[1]]
        tiles.append(tile)

    return tiles



upperTileEncoder = UpperTileEncoder()
lowerTileEncoder = LowerTileEncoder()

epoch = 1;tiles = []
while True:
    print('Beginning epoch ' + str(epoch))

    tiles = [img for img in tiles if random.random() > 1.0 / 100]
    tiles += grab_tiles(100)

    print(str(len(tiles)) + ' tiles in corpus.')

    print('Training upper tile encoder...')
    upperTileEncoder.fit(tiles)
    upperTileEncoder.save()

    print('Training lower tile encoder...')
    chips = upperTileEncoder.encode(tiles)
    lowerTileEncoder.fit(chips)
    lowerTileEncoder.save()

    num_clusters = 10
    points = lowerTileEncoder.encode(chips)
    kmeans = KMeans(n_clusters=num_clusters).fit(points)

    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(tiles[idx])

    for cluster in clusters: print(len(cluster))


    epoch += 1

    print(' ')
