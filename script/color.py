import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from skimage.color import label2rgb
import skimage.data as data
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic, mark_boundaries
import re
import statistics

def color(img, mask):
    im = plt.imread(img)
    mask=plt.imread(mask)

    lesiondimensions=np.where(mask==1.0)

    


    minx= min(lesiondimensions[0])
    maxx= max(lesiondimensions[0])
    miny=min(lesiondimensions[1])
    maxy=max(lesiondimensions[1])

    lesion=im[minx:maxx,miny:maxy]

    segmented_lesion=slic(lesion,n_segments=10, compactness=10, sigma=1, start_label=1)

    all_averages=[]
    for (i, segVal) in enumerate(np.unique(segmented_lesion)):
        mask=np.zeros(lesion.shape[:2],dtype='uint8')
        mask[segmented_lesion==segVal]=255
        merged=cv2.bitwise_and(lesion, lesion, mask = mask)
        
        rgb=0
        
        pixels=0
        
        for x in merged:
            rgb+=np.sum(x,axis=0)
            
        p = np.count_nonzero(merged > 0)
        all_averages.append(rgb)
        
    all_averages = np.divide(all_averages, p)
        
    all_color_float = color_float(all_averages)


    return color_features(all_color_float)




def color_float(color_array):
    final = []
    for rgb in color_array:
        rgb_float = 65536 * rgb[0] + 256 * rgb[1] + rgb[2] # https://quick-adviser.com/can-rgb-value-float/

        final.append(rgb_float)
        
    return final


def color_features(array_):
    array_variance = statistics.variance(array_)
    array_min = min(array_)
    array_max= max(array_)
    array_median= statistics.median(array_)
    array_mean= statistics.median(array_)

    return [array_variance, array_min, array_max, array_median, array_mean]


if __name__ ==  "__main__":
    print(color("../data/raw_data/example_image/ISIC_0001852.jpg", "../data/raw_data/example_segmentation/ISIC_0001852_segmentation.png"))