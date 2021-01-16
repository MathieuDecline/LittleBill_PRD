# -*- coding: utf-8 -*-

# =============================================================================
# Need Utils.py script to run the code
# Document scanning
# =============================================================================



# LIBRARIES
import numpy as np
import cv2
import utlis 
import suivi
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 1) get img 
heightImg = 640
widthImg = 400



pathImage = r'C:\Users\mathi\OneDrive\Bureau\PRO\ECAM Lyon\PRD\Reco\Result\Gallerie\test3.jpg'
#BLANK IMAGE

#utlis.initializeTrackbars()
count = 1

imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
img = cv2.imread(pathImage)
img =cv2.resize(img,(widthImg, heightImg))

#plt.imshow( img)
#plt.show()
# 2) grayscale and edge detector

imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imGray, (5,5), 1)

cv2.imwrite('img.jpg', img)
cv2.imwrite('Gray.jpg', imGray)
cv2.imwrite('Blur.jpg', imgBlur)

#thres = utlis.valTrackbars()
imgThreshold = cv2.Canny(imgBlur, 0, 255)
cv2.imwrite('imgThreshold1.jpg',imgThreshold)
kernel = np.ones((5,5))
imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
imgThreshold = cv2.erode(imgDial, kernel, iterations = 1)


cv2.imwrite('imgThreshold2.jpg', imgThreshold)
cv2.imwrite('imgDial.jpg' , imgDial)


# 3)contours

 ## FIND ALL COUNTOURS
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS

cv2.imwrite('imgContours.jpg',imgContours)

 # FIND THE BIGGEST COUNTOUR
biggest, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
if biggest.size != 0:
        biggest=utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (100, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

cv2.imwrite('imgWarpColored.jpg',imgWarpColored)



#Pre traitement
gray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
cv2.imwrite('Gray.jpg', gray)





# =============================================================================
# NEW IMG GRAY SCALE
# =============================================================================
# reads an input image 


img_Gray = cv2.imread('Gray.jpg',0) 

  
Threshold = suivi.trouverTresholdOptimal(img_Gray)
print('Threshold :' , Threshold)
# Threshold = 100



ret,thresh1 = cv2.threshold(img_Gray,Threshold,255,cv2.THRESH_BINARY)
cv2.imwrite('binar.jpg', thresh1)
print("ret", ret)


#Nouveau PATH
new_PATH =  'binar.jpg'

#Image pour OCR
new_img = Image.open(new_PATH)

#OCR
lecture=pytesseract.image_to_string(new_img)
print(lecture)






