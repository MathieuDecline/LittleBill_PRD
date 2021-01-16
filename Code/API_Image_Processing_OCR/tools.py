# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, Union
import math
from deskew import determine_skew
from typing import Tuple, Union
import numpy as np
import cv2
import math
from deskew import determine_skew

try:
    from PIL import Image
except ImportError:
    import Image

   
    
#!pip install opencv-python
import cv2 

#!pip install pytesseract
import pytesseract
#Tesseract PATH to teseract.exe [Required Folder]
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract.exe'
#see documentation : https://pypi.org/project/pytesseract/

# =============================================================================
# Pre Processing Funtions
# =============================================================================
    
#Read image and transform it into grayscale image
#Input : image
#Output : greyscale_image
def readAndGreyscale(imagePath):
    img_0 = cv2.imread(imagePath)
    img_0=cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('img_gray.jpg', img)
    return img_0

#Find optimal threshold value for binarization
#Input : greyscale_image
#Output : optimal_threshold_value
def trouverTresholdOptimal(greyImage):
    hist = cv2.calcHist([greyImage],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh  
  
#Transform grascale image [value : 0 - 255] into binary image [value : 0 ou 1]
#Input : greyscale_image , threshold_value
#Output : binarized_image
def BinarizationSimple(greyImage, threshold):
    ret,thresh=cv2.threshold(greyImage,threshold,255,cv2.THRESH_BINARY)
    #cv2.imwrite('Results/img_binarized.jpg', thresh)
    cv2.imwrite('optimizedImage.jpg', thresh)
    return thresh

# =============================================================================
# OCR Processing Funtion
# =============================================================================
  
#Print text on image
#Input : greyscale_image
#Output : text_from_image
def printTexteRecu(imageRecu):
    #config = ('-l fra --oem 1 --psm 3')
    text = pytesseract.image_to_string(imageRecu)
    return text

# =============================================================================
# Main Functions
# =============================================================================

#Preprocess the image
#Input : ImageFile
#Output : ImageOptimized
def preprocess(filename):
    print('START_Preprocessing')
    """
    This function will handle image processing of the input.
    """
    #PATH to image
    PATH_IMAGE = filename
    #Pipeline
    greyImage =readAndGreyscale(PATH_IMAGE)
    threshold = trouverTresholdOptimal(greyImage)
    binarizedImage =BinarizationSimple(greyImage, threshold)
    print('END_Preprocessing')
    return binarizedImage

 
#Print text on image
#Input : greyscale_image
#Output : text_from_image
def ocr(optimizedImage):
    print('START_OCR') 
    """
    This function will handle the core OCR processing of images.
    """
    text = printTexteRecu(optimizedImage)
    print ('END_OCR')
    return text




