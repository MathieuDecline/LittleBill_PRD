# -*- coding: utf-8 -*-

from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
from typing import Tuple, Union
import math
from deskew import determine_skew

#lis l image couleur et la transforme en niveaux de gris
def readAndGreyscale(imagePath):
    img_0 = cv2.imread(imagePath)
    img=cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    return img

#Redressement de l'image
def rotate(
        image: np.ndarray, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    angle = determine_skew(image)
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)



#redimensionnement de l image
def resize(image, scale_percent):#percent by which the image is resized
    
    #calculate the 50 percent of original dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    # resize image
    img = cv2.resize(image, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img

#Recuperation de l histogramme
#Entree : image grayscale
#Sortie : histogrammee 
def Histogramme(greyImage):
    img = cv2.imread(greyImage,0) 
    histr = cv2.calcHist([greyImage],[0],None,[256],[0,256]) 
    plt.plot(histr) 
    return plt.show() 


#Recherche du seuil minimum optimal selon l histogramme
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

#traitement blur et binarization otsu
def otsuThresholdAndBlur(greyImage):
    blur = cv2.GaussianBlur(greyImage,(5,5),0)
    ret,otsu=cv2.threshold(blur,trouverTresholdOptimal(greyImage),255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsu


def BinarizationSimple(greyImage):
    ret,thresh=cv2.threshold(greyImage,trouverTresholdOptimal(greyImage),255,cv2.THRESH_BINARY)
    return thresh
    
#traitement pour netteté 
def nettete(binarizedImage):
    blur = cv2.GaussianBlur(binarizedImage,(15,15),0)#parametres a modifier selon l image
    return cv2.addWeighted(binarizedImage,2.0,blur,-0.8,0)#parametres a modifier


def printTexteRecu(imageRecu):
    config = ('-l fra --oem 1 --psm 3')
    print (pytesseract.image_to_string(imageRecu, config=config))

"""
#exemple utilisation
img=readAndGreyscale('mat.jpg')
img=rotate(img,(0,0,0))
img=otsuThresholdAndBlur(img)
img=nettete(img)
cv2.imwrite('nouvelleImage.jpg',img)
printTexteRecu(img)

#Histo
Histogramme('mat_gray.jpg')


"""

