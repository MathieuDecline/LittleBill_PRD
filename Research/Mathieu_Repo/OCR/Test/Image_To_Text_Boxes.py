# -*- coding: utf-8 -*-

#!pip install opencv-python
import cv2 

#!pip install pytesseract
import pytesseract
#Tesseract PATH to teseract.exe [Required Folder]
#see documentation : https://pypi.org/project/pytesseract/
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import numpy as np
from PIL import ImageGrab
import time


###Detecting Characters
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
        cv2.imwrite(Output_Image, img)

###Detecting Words - Plot Boxes
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
                
    cv2.imwrite(Output_Image, img)
    
# =============================================================================
    
print('START')

InputImage = r'C:\Users\mathi\OneDrive\Bureau\PRO\ECAM Lyon\PRD\Reco\Result\Gallerie\X51008123476.jpg' 
OutputWords = 'ResultWords.jpg'
OutputCharacters = 'ResultCharacter.jpg'

Img_Boxes_Words_Text(InputImage, OutputWords)
Img_Boxes_Characters_Text(InputImage, OutputCharacters)

print('END')
