from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt
image = cv2.imread('watershed_coins_01.jpg')
shifted = cv2.pyrMeanShiftFiltering(image,21,51)

gray = cv2.cvtColor(shifted,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D,indices=False,min_distance=20,labels=thresh)

markers = ndimage.label(localMax,structure=np.ones((3,3)))[0]
labels = watershed(-D,markers,mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels))-1))
plt.imshow(D)
plt.show()
plt.imshow(localMax)
plt.show()
plt.imshow(markers)
plt.show()
plt.imshow(labels)
plt.show()