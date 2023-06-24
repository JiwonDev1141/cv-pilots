import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define our imshow function
def imshow(title = "image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


image = cv2.imread('images/scan.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


cv2.drawContours(image, contours, -1, (0,255,0), thickness= 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))


sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

for cnt in sorted_contours:
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)

    if len(approx) == 4:
        break

print("Our 4 corner points are:")
print(approx)

inputPts = np.float32(approx)

outputPts = np.float32([[0,0], [0,800], [500,800], [500,0]])

M = cv2.getPerspectiveTransform(inputPts, outputPts)

dst = cv2.warpPerspective(image, M, (500,800))

imshow('Perspective', dst)