import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('water_coins.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('img',thresh)
cv2.imshow('original',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# noise removal
kernel = np.ones((3,3),dtype=np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
plt.imshow(opening,'gray')
plt.show()
# sure background area
sure_bg = cv2.dilate(opening,kernel=kernel,iterations=3)
plt.imshow(sure_bg,'gray')
plt.show()
# finding sure foreground data
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
plt.imshow(dist_transform)
plt.show()
plt.imshow(sure_fg)
plt.show()
# Finding the unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow('unknown',unknown)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Marker Labelling
ret, markers = cv2.connectedComponents(sure_fg)
plt.imshow(markers)
plt.show()

# Add 1 to all labels so that background is not 0 but 1
markers += 1
# Now mark the region of unknown as 0
markers[unknown == 255] = 0
plt.imshow(markers)
plt.show()
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
plt.imshow(img)
plt.show()
plt.imshow(img)
plt.show()
plt.imshow(markers)
plt.show()
