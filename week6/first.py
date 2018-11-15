import skimage
from sklearn.cluster import KMeans
import numpy as np
from skimage.measure import compare_psnr
import matplotlib.pyplot as plt
from skimage import novice


image = skimage.io.imread('parrots.jpg')
image = skimage.img_as_float(image)

picture = novice.open('parrots.jpg')
width = picture.width
height = picture.height

def show_img(pixels):
    for y in range(0, height):
        for x in range(0, width):
            image[y][x] = pixels[x + width * y]
    plt.imshow(image)
    plt.show()

pixels = []
for row in image:
    for pixel in row:
        pixels.append(pixel)
pixels = np.array(pixels)
show_img(pixels)

for n_clusters in range (1, 21):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    kmeans.fit(np.array(pixels))
    X_pred = kmeans.predict(pixels)
    centers = kmeans.cluster_centers_
    X_medians = pixels.copy()
    for idx in range(0, len(X_pred)):
        X_medians[idx] = centers[X_pred[idx]]
    show_img(X_medians)
    print(n_clusters, compare_psnr(pixels, X_medians))

