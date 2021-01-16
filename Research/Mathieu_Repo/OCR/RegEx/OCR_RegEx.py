# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

#!pip install re
import re

#!pip install opencv-python
import cv2 

import numpy as np

#!pip install pytesseract
import pytesseract
#Tesseract PATH to teseract.exe [Required Folder]
#see documentation : https://pypi.org/project/pytesseract/
pytesseract.pytesseract.tesseract_cmd=r'PATH_TO_TESSERACT\tesseract.exe'

#Convert RGB_Image into Greyscale_Image
#Input : PATH to RGB_Image
#Output : Greyscale_Image
def greyscale(PATH):
    img_0 = cv2.imread(PATH)
    img=cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    return img

#Convert every pixel into Black or White depending on Threshold
#Input : Greyscale_Image, Threshold_Value
#Output : Binarized_Image (Only Black and White pixels)
def binarization(greyImage, Threshold):
    ret,thresh=cv2.threshold(greyImage,Threshold,255,cv2.THRESH_BINARY)
    return thresh 

#Convert picture into text from the picture
#Input : Any Image With Text (Our image has been preprocessed)
#Output : Text from image
def ocr(InputImage):
    lang = '_ _'
    oem = '--oem1'
    psm = '--psm3'
    config = (lang + oem + psm)
    text = pytesseract.image_to_string(InputImage)
    return text
 
#Find the optimal threshold (pixel distribution between 0 - 255)
#Input : Grey_Image 
#Output : ThresholdValue
def optimalTreshold(greyImage):
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

#Find pattern (RegEx) associated to a word
#Input : Text_to_search
#Output : List of matches (matching word)
def characterRegex(TEXT):
    pattern = re.compile(r'\w') 
    matches = pattern.finditer(TEXT)
    lst_matches = []
    for matches in matches:
        #print(matches)
        word = matches[0]
        lst_matches.append(word)   
    return 'Text:', lst_matches

#Find pattern (RegEx) associated a price
#Input : Text_to_search
#Output : List of matches (matching prices)
def priceRegex(TEXT):
    pattern = re.compile(r'\.?\d+\.\d\d\.?') 
    matches = pattern.finditer(TEXT)
    lst_matches = []
    for matches in matches:
        #print(matches)
        word = matches[0]
        lst_matches.append(word)   
    return 'Prices:', lst_matches

#Extract maximum value and its position from list
#Input : LIST
#Output : maximum_value, position_of_maximum_value_in_list
def maximum(LIST):
    maxVal = 0
    Position=0
    val=0
    for i in range (0,len(LIST),1):
        val=float(LIST[i])
        if (val>=maxVal):
            maxVal=val
            Position=i
        else : 
            maxVal=maxVal
            Position=Position
    return maxVal , Position

#Find the threshold corresponding to the maximum of WORD matching
#Input : PATH to Image
#Output : Maximum_Matches , Text_from_image , AllText , AllSizes, AllPositions [All : For all thresholds (256)]
def thresholdMaxContent(PATH):
    greyImg = greyscale(PATH)
    content = []
    size = []
    pos = []  
    for thresh in range (0,255, 1):
        #print (thresh)
        imgBinar = binarization(greyImg, thresh)
        ocr_result = ocr(imgBinar)
        ocr_size = len(characterRegex(ocr_result)[1])
        ocr_position = thresh 
        #print('Result : ', OCR_Result)
        print('LEN : ', ocr_size)
        print('Position : ', ocr_position)
        #print(OCR_Result)
        content.append(ocr_result)
        size.append(ocr_size)
        pos.append(ocr_position)     
    return maximum(size) , content[maximum(size)[1]] , content, size, pos

#Find the threshold corresponding to the maximum of PRICE matching
#Input : PATH to Image
#Output : Maximum_Matches , Text_from_image , AllText , AllSizes, AllPositions [All : For all thresholds (256)]
def thresholdMaxPrices(PATH):
    greyImg = greyscale(PATH)
    content = []
    size = []
    pos = []  
    for thresh in range (0,255, 1):
        #print (thresh)
        imgBinar = binarization(greyImg, thresh)
        ocr_result = ocr(imgBinar)
        ocr_size = len(priceRegex(ocr_result)[1])
        ocr_position = thresh 
        #print('Result : ', OCR_Result)
        print('LEN : ', ocr_size)
        print('Position : ', ocr_position)
        #print(ocr_result)
        content.append(ocr_result)
        size.append(ocr_size)
        pos.append(ocr_position)  
    return maximum(size) , content[maximum(size)[1]] , content, size, pos

#Find pattern (RegEx) associated to a TOTAL and then look for the biggest following price
#Input : text_to_search
#Output : maximum value of matches 
def totalRegex(Text):
    total_pattern = re.compile(r'([Tt][Oo][Tt][Aa][Ll]|[Bb][Aa][Ll][Aa][Nn][Cc][Ee]|[Aa][Mm][Oo][Nn][Tt]|[Tt][Oo][Tt]|[Dd][Uu][Ee]|[Aa][Mm][Tt])\.?') 
    total_matches = total_pattern.finditer(Text)
    price_pattern = re.compile(r'\.?\d+\.\d\d\.?')
    if (total_pattern.search(Text) != None):
        m= total_pattern.search(Text)
        price_matches = price_pattern.finditer(Text[m.span()[1]:])
        lst_total=[]
        for matches in price_matches:
            #print(matches)
            word = matches[0]
            lst_total.append(word)
        if (len(lst_total)!=0) :
            return 'TOTAL : ' , maximum(lst_total)
        else :
            return 'TOTAL : ' , lst_total

#Find pattern (RegEx) associated to a DATE 
#Input : text_to_search
#Output : list of matches 
def dateRegex(Text):
    pattern = re.compile(r'(\b(\d\d|\d|)[/](\d\d|\d)[/](\d\d\d\d|\d\d)|\b(\d\d|\d|)[-](\d\d|\d)[-](\d\d\d\d|\d\d))') 
    date_matches = pattern.finditer(Text)
    lst_date =[]
    for matches in date_matches:
        #print(matches)
        word = matches[0]
        lst_date.append(word)   
    return 'DATE:', lst_date

#Find pattern (RegEx) associated to a TVA and then look for the following price
#Input : text_to_search
#Output : list of matches 
def TVARegex(Text):
    TVA_pattern = re.compile(r'[Tt][Vv][Aa]|[Tt][Aa][Xx]') 
    price_pattern = re.compile(r'\.?(\d+\.\d\d|\d+\.?)\.?')
    lst_TVA=[]
    if (TVA_pattern.search(Text)!= None):
        m= TVA_pattern.search(Text)
        price_matches = price_pattern.finditer(Text[m.span()[1]:])
        for matches in price_matches:
            #print(matches)
            word = matches[0]
            lst_TVA.append(word)
    return 'TVA : ' , lst_TVA

#Find pattern (RegEx) associated to a currency 
#Input : text_to_search
#Output : One currency matching
def deviseRegex(Text):
    devise_pattern = re.compile(r'[$€]') 
    Devise_matches = devise_pattern.finditer(Text)
    lst_devise=[]
    for matches in Devise_matches:
        #print(matches)
        word = matches[0]
        lst_devise = word
    return 'Devise : ' , lst_devise

   
# =============================================================================
# CONSOLE    
# =============================================================================

print('START')

#Path To Image
PATH = r'PATH_TO_IMAGE\ImageName.jpg'

result = thresholdMaxPrices(PATH)
print(result[0])
print (result[1])

#Patterns
resultRegexTotal = totalRegex(result[1])
resultRegexDate = dateRegex(result[1])
resultRegexTVA = TVARegex(result[1])
resultRegexDevise = deviseRegex(result[1])

print('Total : ', resultRegexTotal[1][0])
# print('Date : ', result_Regex_Date[1][0])
# print('TVA : ', result_Regex_TVA[1][0])
# print('Devise : ', result_Regex_Devise[1][0])
    
#Plotting matching_frequency = f(position) [HIST]
# plt.plot(Result[4], Result[3])
# plt.show()

print('END')
