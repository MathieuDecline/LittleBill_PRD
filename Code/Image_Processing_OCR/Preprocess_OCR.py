# -*- coding: utf-8 -*-

#Import Libraries
from PIL import Image
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

#!pip install opencv-python
import cv2 

#!pip install pytesseract
import pytesseract
#Tesseract PATH to teseract.exe [Required Folder]
#see documentation : https://pypi.org/project/pytesseract/
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#lis l image couleur et la transforme en niveaux de gris
#Input : image
#Output : greyscale_image
def readAndGreyscale(imagePath):
    img_0 = cv2.imread(imagePath)
    img=cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('Outputs/img_gray.jpg', img)
    return img

#Recherche du seuil minimum optimal selon l histogramme
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
  
#Conversion de greyscale_image [valeurs : 0 - 255] en image_binaire [valeurs : 0 ou 1]
#Input : greyscale_image , threshold_value
#Output : binarized_image
def BinarizationSimple(greyImage, threshold):
    ret,thresh=cv2.threshold(greyImage,threshold,255,cv2.THRESH_BINARY)
    print(trouverTresholdOptimal(greyImage))
    cv2.imwrite('Outputs/img_binarized.jpg', thresh)
    return thresh

#Print le texte present sur une image
#Input : greyscale_image
#Output : text_from_image
def printTexteRecu(imageRecu):
    #config = ('-l fra --oem 1 --psm 3')
    text = pytesseract.image_to_string(imageRecu)
    text_file = open('/Outputs/Text_from_image.txt', 'w+')
    text_file.write(text)
    print (text)

#Recuperation de l histogramme
#Input : image_greyscale
#Output : histogram_from_img
def histogramme(greyImage): 
    image=greyImage
    histr = cv2.calcHist([image],[0],None,[256],[0,256]) 
    plt.plot(histr) 
    plt.show() 
    return histr

#Redressement de l'image
#Input : image
#Output : image_rotated
def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

#Greyscale et rotation de l image
#Input : image
#Output : greyscale_image_rotated
def openGreyRotate(imagePath):
      imgColor = cv2.imread(imagePath)
      imgGrey=cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
      angle = determine_skew(imgGrey)
      rotated = rotate(imgColor, angle+90, (0, 0, 0))
      rotatedGrey=cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
      return rotatedGrey

#redimensionnement de l image
#Input : image , pourcentage
#Output : resized_image
def resize(image, scale_percent):#percent by which the image is resized
    #calculate the 50 percent of original dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    # resize image
    img = cv2.resize(image, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img

    
#traitement blur et binarization otsu
#Input : greyscale_image
#Output : optimal_threshold_value
def otsuThresholdAndBlur(greyImage):
    blur = cv2.GaussianBlur(greyImage,(5,5),0)
    ret,otsu=cv2.threshold(blur,trouverTresholdOptimal(greyImage),255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsu

#traitement pour nettete
#Input : binarized_image
#Output : image_nette
def nettete(binarizedImage):
    blur = cv2.GaussianBlur(binarizedImage,(15,15),0)#parametres a modifier selon l image
    return cv2.addWeighted(binarizedImage,2.0,blur,-0.8,0)#parametres a modifier

#Grossir les caractères
#Input : greyscale_image
#Output : dilated_image
def Dilatation(image):
        ImOut = image.copy() # copie de l'image
        width,height = image.size # taille de l'image
        for indexH in range(height): # parcours des pixels en colonne
                for indexW in range(width): # parcours des pixels en ligne
                        dilate = False
                        if indexW > 0 and image.getpixel((indexW - 1, indexH)) == 0: # on regarde le pixel de gauche
                                dilate = True
                        elif indexW < width - 1 and image.getpixel((indexW + 1, indexH)) == 0: # on regarde le pixel de droite
                                dilate = True
                        elif indexH > 0 and image.getpixel((indexW, indexH - 1)) == 0: # on regarde le pixel du haut
                                dilate = True
                        elif indexH < height - 1 and image.getpixel((indexW, indexH + 1)) == 0: # on regarde le pixel du bas
                                dilate = True
                                
                        if dilate == True:
                                ImOut.putpixel((indexW, indexH), 0) # le pixel devient noir si un de ses voisins est noir
        return ImOut
    

#Detecting Characters - Plot Boxes
#Input : InputImage , OutputImage
#Output : ImageFile 
def Img_Boxes_Characters_Text(Input_Image, Output_Image):
    img = cv2.imread(Input_Image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hImg,WImg,_ = img.shape
    boxes =pytesseract.image_to_boxes(img)
    for b in boxes.splitlines():
        b = b.split(' ')
        x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),1)
        cv2.putText(img,b[0], (x,hImg-y+10),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)
        cv2.imwrite('/Results/'+ Output_Image, img)

#Detecting Words - Plot Boxes
#Input : InputImage , OutputImage
#Output : ImageFile 
def Img_Boxes_Words_Text(Input_Image, Output_Image):
    img = cv2.imread(Input_Image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hImg,WImg,_ = img.shape
    boxes =pytesseract.image_to_data(img)
    for x,b in enumerate(boxes.splitlines()):
        if x != 0:
            b = b.split()
            if len(b)==12 :
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),1)
                cv2.putText(img,b[11], (x,y+10),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)          
    cv2.imwrite('/Results/'+ Output_Image, img)
    return boxes


# =============================================================================
# ================================== CONSOLE ==================================
# readAndGreyscale(imagePath)
# printTexteRecu(imageRecu)
# trouverTresholdOptimal(greyImage)
# BinarizationSimple(greyImage, threshold)
# =============================================================================
# histogramme(greyImage)
# rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]])
# openGreyRotate(imagePath)
# resize(image, scale_percent)
# otsuThresholdAndBlur(greyImage)
# nettete(binarizedImage)
# Dilatation(image)
# =============================================================================
# Img_Boxes_Characters_Text(Input_Image, Output_Image)
# Img_Boxes_Words_Text(Input_Image, Output_Image)
# =============================================================================
print('START')

#PATH to image
PATH_IMAGE = r'Image_Receipt.jpg'

#Pipeline
greyImage =readAndGreyscale(PATH_IMAGE)
threshold = trouverTresholdOptimal(greyImage)
binarizedImage =BinarizationSimple(greyImage, threshold)
printTexteRecu(binarizedImage)

#Threshold_hist
histogramme(greyImage)

#Tesseract functions
Img_Boxes_Words_Text(PATH_IMAGE, 'img_boxes_words.jpg')
Img_Boxes_Characters_Text(PATH_IMAGE, 'img_boxes_sentences.jpg')


print('END')
