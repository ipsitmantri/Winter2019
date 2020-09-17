from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="watershed_coins_01.jpg")
#args = vars(ap.parse_args())

# load the image and perform the pyramid mean shift filtering
# to aid the thresholding step
image = cv2.imread("watershed_coins_01.jpg")
shifted = cv2.pyrMeanShiftFiltering(image,21,51)
cv2.imshow("input",image)
cv2.imshow("shifted",shifted)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(shifted,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("thresh",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("[INFO] {} unique contours found".format(len(cnts)))

for i,c in enumerate(cnts):
    ((x,y),_) = cv2.minEnclosingCircle(c)
    cv2.putText(image,"#{}".format(i+1),(int(x) - 10,int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.6,
                (0,0,255),2)
    cv2.drawContours(image,[c],-1,(0,255,0),2)

cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()