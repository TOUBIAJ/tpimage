

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def region_growth(image, seeds, threshold):
    region = {}
    frontier = set()
    label = 1

    for seed in seeds:
        region[seed] = label
        frontier.add(seed)

    while frontier:
        current_pixel = frontier.pop()

        for neighbor in [(current_pixel[0] + 1, current_pixel[1]),
                         (current_pixel[0] - 1, current_pixel[1]),
                         (current_pixel[0], current_pixel[1] + 1),
                         (current_pixel[0], current_pixel[1] - 1)]:
            if 0 <= neighbor[0] < image.shape[0] and 0 <= neighbor[1] < image.shape[1]:
                if neighbor not in region and abs(int(image[current_pixel[0], current_pixel[1]]) - int(image[neighbor[0], neighbor[1]])) <= threshold:
                    region[neighbor] = label
                    frontier.add(neighbor)
    
    return region

def color_regions(image, regions):
    colors = {}
    colored_image = np.zeros(image.shape + (3,), dtype=np.uint8)

    for region_label in regions.values():
        colors[region_label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if (y, x) in regions:
                colored_image[y, x] = colors[regions[(y, x)]]

    return colored_image

image = cv2.imread('/home/joe/Downloads/Images/lena.png', cv2.IMREAD_GRAYSCALE)

seed_points = [(100, 100), (200, 200),(50,50),(40,40)]  # Liste des germes
similarity_threshold = 8

resulting_region = region_growth(image, seed_points, similarity_threshold)

colored_image = color_regions(image, resulting_region)

plt.imshow(colored_image)
plt.title('Image segmentée avec plusieurs germes et couleurs différentes pour chaque région')
plt.axis('off')
plt.show()



