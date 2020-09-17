import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF
img = cv2.imread('cells.png')
cv2.imshow('cells',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
modified = -np.log(img/255)
print(modified.shape)
plt.imshow(modified,'gray')
plt.show()
model = NMF(n_components=2,init='random',random_state=0)
W = []
H = []
for i in range(3):
    Wi = model.fit_transform(modified[:,:,i])
    Hi = model.components_
    W.append(Wi)
    H.append(Hi)
