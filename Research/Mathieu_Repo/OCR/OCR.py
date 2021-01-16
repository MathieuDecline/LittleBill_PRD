# -*- coding: utf-8 -*-


#!pip install opencv-python
import cv2 

#!pip install pytesseract
import pytesseract
#Tesseract PATH to teseract.exe [Required Folder]
#see documentation : https://pypi.org/project/pytesseract/
pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'


#Convert Picture into text file
#Input : Image
#Output : Text
def ImageToText(Input_image, Output_Text):
    img = cv2.imread(PATH)
    #Preprocessing grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text=pytesseract.image_to_string(gray_img)
    text_file = open(Output_Text, 'w+')
    text_file.write(text)
    return text

# =============================================================================

print('START')

#PATH to image 
PATH = r'C:\Users\mathi\OneDrive\Bureau\PRO\ECAM Lyon\PRD\Reco\Result\Gallerie\3.jpg'
TextFile = 'TextFromImage.txt'

ImageToText(PATH,TextFile)

print('END')
