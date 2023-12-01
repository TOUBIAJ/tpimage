
import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('/home/joe/Downloads/Images/lena.png', cv2.IMREAD_GRAYSCALE)
histogram = np.zeros(256)


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        pixel_value = image[i, j]
        histogram[pixel_value] += 1


histogram_normalized = histogram / float(image.size)
plt.subplot(121), plt.plot(histogram), plt.title('Histogramme d\'origine')
plt.subplot(122), plt.plot(histogram_normalized), plt.title('Histogramme normalisé')
plt.show()




Nmin = np.inf
Nmax = -np.inf

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        pixel_value = image[i, j]
        Nmin = min(Nmin, pixel_value)
        Nmax = max(Nmax, pixel_value)

a = 0
b = 255

image_stretched = ((b - a) / (Nmax - Nmin)) * (image - Nmin) + a
image_stretched = np.clip(image_stretched, 0, 255).astype(np.uint8)
new_size = (300, 300)
image_resized = cv2.resize(image, new_size)
image_stretched_resized = cv2.resize(image_stretched, new_size)

hist_original = np.zeros(256, dtype=int)
hist_etiree = np.zeros(256, dtype=int)


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        hist_original[image[i, j]] += 1


for i in range(image_stretched.shape[0]):
    for j in range(image_stretched.shape[1]):
        hist_etiree[image_stretched[i, j]] += 1

hist_original_image = np.zeros((220, 256), dtype=np.uint8)
hist_etiree_image = np.zeros((220, 256), dtype=np.uint8)


hist_original_normalized = hist_original * (hist_original_image.shape[0] / np.max(hist_original))
hist_etiree_normalized = hist_etiree * (hist_etiree_image.shape[0] / np.max(hist_etiree))

for i in range(256):
    cv2.line(hist_original_image, (i, int(220)), (i, int(220 - hist_original_normalized[i])), 255)
    cv2.line(hist_etiree_image, (i, int(220)), (i, int(220 - hist_etiree_normalized[i])), 255)

cv2.imshow('Image Resize', image_resized)
cv2.imshow('Image Stretched Resize', image_stretched_resized)
cv2.imshow('Hist Original', hist_original_image)
cv2.imshow('Hist Etiree', hist_etiree_image)



image_equalized = cv2.equalizeHist(image)

plt.figure(figsize=(10, 5))

plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title('Image Originale')

plt.subplot(222)
plt.imshow(image_equalized, cmap='gray')
plt.title('Image Égalisée')


hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([image_equalized], [0], None, [256], [0, 256])
hist_original_normalized = hist_original * (220 / np.max(hist_original))
hist_equalized_normalized = hist_equalized * (220 / np.max(hist_equalized))


plt.subplot(223)
plt.plot(hist_original_normalized, color='black')
plt.title('Histogramme Image Originale')

plt.subplot(224)
plt.plot(hist_equalized_normalized, color='black')
plt.title('Histogramme Image Égalisée')

plt.tight_layout()
plt.show()




filtre = np.array([[0, 1/7, 0],
                   [1/9, 1/9, 1/9],
                   [0, 1/7, 0]])


def filtrage(image, filtre):
    hauteur, largeur = image.shape
    image_filtree = np.zeros_like(image)

    
    for i in range(1, hauteur - 1):
        for j in range(1, largeur - 1):
            fenetre = image[i-1:i+2, j-1:j+2]
            somme_ponderee = np.sum(fenetre * filtre)
            image_filtree[i, j] = somme_ponderee

    return image_filtree




filtre = np.array([[0, 1/7, 0],
                   [1/9, 1/9, 1/9],
                   [0, 1/7, 0]])


def filtrage(image, filtre):
    hauteur, largeur = image.shape
    image_filtree = np.zeros_like(image)

    
    for i in range(1, hauteur - 1):
        for j in range(1, largeur - 1):
            fenetre = image[i-1:i+2, j-1:j+2]
            somme_ponderee = np.sum(fenetre * filtre)
            image_filtree[i, j] = somme_ponderee

    return image_filtree


image_filtree = filtrage(image, filtre)


new_size = (300, 300)
image_resized = cv2.resize(image, new_size)
image_filtree_resized = cv2.resize(image_filtree, new_size)

font_color = (0, 0, 255)
cv2.putText(image_resized, 'Image Originale', (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, font_color, 2, cv2.LINE_AA)
cv2.putText(image_filtree_resized, 'Image Filtree', (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, font_color, 2, cv2.LINE_AA)

cv2.imshow('image_original',image_resized)
cv2.imshow('image_filtre',image_filtree_resized)


hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_filtree = cv2.calcHist([image_filtree], [0], None, [256], [0, 256])


hist_height = 150
hist_original_image = np.zeros((hist_height, 256), dtype=np.uint8)
hist_filtree_image = np.zeros((hist_height, 256), dtype=np.uint8)


hist_original_normalized = hist_original * (hist_original_image.shape[0] / np.max(hist_original))
hist_filtree_normalized = hist_filtree * (hist_filtree_image.shape[0] / np.max(hist_filtree))


for i in range(256):
    cv2.line(hist_original_image, (i, int(hist_height)), (i, int(hist_height - hist_original_normalized[i])), 255)
    cv2.line(hist_filtree_image, (i, int(hist_height)), (i, int(hist_height - hist_filtree_normalized[i])), 255)


cv2.imshow('hist_original',hist_original_image)
cv2.imshow('hist_filtre',hist_filtree_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
