import numpy as np
from skimage import filters, morphology
import matplotlib.pyplot as plt
import os
import math
import csv
import pandas as pd



def smoothness_all():
    """
        Calculates the Circularity from area and perimeter of all given fetures in features.csv
    """
    data= pd.read_csv("../data/features/features.csv")

    features = []
    for _, row in data.iterrows():    
        features.append([row["id"], int(row["perimeter"])**2/(4*math.pi*int(row["area"]))])

    features = pd.DataFrame(features, columns=["id", "Circularity"])
    return features

def calc_features(img):
    """
        Calculates features for a given black and white image.
        Fetures are area and perimeter.
    """

    mask=plt.imread(img)
    # create brush
    struct_el = morphology.disk(1)

    # Use this "brush" to erode the image - eat away at the borders

    mask_eroded = morphology.binary_erosion(mask, struct_el)

    

    # calculate area

    area= np.sum(mask)

    # calculate perimeter
    
    # Subtract the two masks from each other to get the border/perimeter
    image_perimeter = mask - mask_eroded

    perimeter = np.sum(image_perimeter)

    return perimeter, area

def circularity(features):
    """
    Calculate circularity from a list of features (perimeter, area)
    """
    return features[0]**2/(4*math.pi*features[1])

if __name__ == '__main__':  
    img = '../data/raw_data/example_segmentation/ISIC_0001769_segmentation.png'
    print(circularity(calc_features(img)))