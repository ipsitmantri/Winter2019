import cv2
import numpy as np
from matplotlib import pyplot as plt
original_img = cv2.imread('binary_mask.png',0)
edges = cv2.Canny(original_img,100,100)
plt.subplot(1,2,1),plt.imshow(original_img,'gray'),plt.title('original')
plt.subplot(1,2,2),plt.imshow(edges,'gray'),plt.title('edges')
plt.show()
boundary_class = original_img.copy()
boundary_class[:,:] = np.uint8(0)
boundary_class[edges == np.uint8(255)] = np.uint8(255)
plt.imshow(boundary_class,'gray')
plt.show()
for i in range(boundary_class.shape[0]):
    for j in range(boundary_class.shape[1]):
        if edges[i,j] == np.uint8(255):
            try :
                boundary_class[i, j - 1] = np.uint8(255)
                boundary_class[i, j + 1] = np.uint8(255)
                boundary_class[i - 1, j] = np.uint8(255)
                boundary_class[i + 1, j] = np.uint8(255)
            except Exception as e:
                print('pass')
            # if i != 0 and j !=0 and i!=999 and j!=999:
            #     boundary_class[i,j-1] = np.uint8(255)
            #     boundary_class[i,j+1] = np.uint8(255)
            #     boundary_class[i-1,j] = np.uint8(255)
            #     boundary_class[i+1,j] = np.uint8(255)
            # if i == 0 and j!= 0:
            #     boundary_class[i, j - 1] = np.uint8(255)
            #     boundary_class[i, j + 1] = np.uint8(255)
            #     boundary_class[i + 1, j] = np.uint8(255)
            # if i !=0 and j == 0:
            #     boundary_class[i, j + 1] = np.uint8(255)
            #     boundary_class[i - 1, j] = np.uint8(255)
            #     boundary_class[i + 1, j] = np.uint8(255)
            # if i == 0 and j == 0:
            #     boundary_class[i, j + 1] = np.uint8(255)
            #     boundary_class[i + 1, j] = np.uint8(255)
plt.imshow(boundary_class,'gray'),plt.title('boundary class')
plt.show()
inside_class = original_img.copy()
inside_class[boundary_class == np.uint8(255)] = np.uint8(0)
plt.subplot(1,2,1),plt.imshow(inside_class,'gray'),plt.title('inside_class')
plt.subplot(1,2,2),plt.imshow(original_img,'gray'),plt.title('original')
plt.show()
ternary_mask = np.zeros_like(original_img)
ternary_mask[boundary_class==np.uint8(255)] = 1
ternary_mask[inside_class==np.uint8(255)] = 0
outside_class = original_img.copy()
outside_class = cv2.bitwise_not(outside_class)
outside_class[boundary_class==np.uint8(255)] = np.uint8(0)
plt.subplot(1,2,1),plt.imshow(outside_class,'gray'),plt.title('outside_class')
plt.subplot(1,2,2),plt.imshow(original_img,'gray'),plt.title('original')
plt.show()
ternary_mask[outside_class==np.uint8(255)] = 2
print(np.unique(ternary_mask))
