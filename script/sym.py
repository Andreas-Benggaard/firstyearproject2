import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from numpy import asarray
from skimage import transform




def find_center(img):
    M = cv2.moments(img)

    # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])


    center = cX, cY
    return center

# Found by Erling Amundsen
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (find_center(image))

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def flip_image(image):
    right_side = image[:,find_center(image)[0]:]
    left_side = cv2.flip(image[:,:find_center(image)[0]],1)

    if right_side.shape[1]>left_side.shape[1]:
        right_side = right_side[:,:left_side.shape[1]]
    else:
        left_side = left_side[:,:right_side.shape[1]]

    up_side = image[:find_center(image)[1],:]
    down_side = cv2.flip(image[find_center(image)[1]:,:],0)
    if up_side.shape[0]>down_side.shape[0]:
        diff = int(np.abs(up_side.shape[0]-down_side.shape[0]))
        up_side = up_side[diff:,:]
    else:
        diff = int(np.abs(up_side.shape[0]-down_side.shape[0]))
        down_side = down_side[diff:,:]
    dif_x,dif_y = int(np.sum(right_side-left_side)),np.abs(int(np.sum(up_side-down_side)))
    return (dif_x+dif_y)//2

def symmetry(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = rotate_bound(img, 0)
    rot_img = rotate_bound(img, 45)
    return (flip_image(img)+flip_image(rot_img))//2