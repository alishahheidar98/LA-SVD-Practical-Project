import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

plt.style.use('classic')
img = Image.open('C:\\Users\\user\\Desktop\\BW-using-curves.jpg')

image = img.convert('LA')

imageMatrix = np.array(list(image.getdata(band=0)), float)

imageMatrix.shape = (image.size[1], image.size[0])

plt.figure(figsize=(9, 6))
plt.imshow(imageMatrix, cmap='gray')
plt.show()

U, Sigma, V = np.linalg.svd(imageMatrix)
imageMatrix.shape = (999, 1498)
U.shape = (999, 999)
Sigma.shape = (999,)
V.shape = (1498, 1498)

decomposition = np.matrix(U[:, :2]) * np.diag(Sigma[:2]) * np.matrix(V[:2, :])
plt.imshow(decomposition, cmap='gray')
plt.show()

for i in [0, 5, 10, 15, 20, 30, 50, 300, 700]:
    decomposition = np.matrix(U[:, :i]) * np.diag(Sigma[:i]) * np.matrix(V[:i, :])
    plt.imshow(decomposition, cmap='gray')
    title = "k = %s" % i
    plt.title(title)
    plt.show()

plt.stem(Sigma)
plt.show()
