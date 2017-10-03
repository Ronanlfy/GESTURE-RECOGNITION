import cv2
import numpy as np
import copy
import math
from appscript import app

def removeBG(img):
    fgmask = bgModel.apply(img)
    
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(img, img, mask=fgmask)
    return res

def fingernumber(cnt,drawing):
    hull = cv2.convexHull(cnt,returnPoints = False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img,far,1,[0,0,255],-1)
        #dist = cv2.pointPolygonTest(cnt,far,True)
        cv2.line(crop_img,start,end,[0,255,0],2)
#cv2.circle(crop_img,far,5,[0,0,255],-1)
    return count_defects



##main program

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, img = cap.read()
    
    bgModel = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    img = cv2.bilateralFilter(img, 5, 50, 100)
    ##draw rectangle in the original image
    cv2.rectangle(img,(50,50),(400,600),(0,255,0),0)
    
    crop_img = img[50:600, 50:400]
 
    img = cv2.flip(img, 1)
  

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    value = (41, 41)
    blurred = cv2.GaussianBlur(gray, value, 0)
    _, thresh1 = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ##  cv2.imshow('Thresholded', thresh1)
    
    crop_img = removeBG(crop_img)

##check the version of python
    (version, _, _) = cv2.__version__.split('.')

    if version is '3':
        ##find the contour, NONE stores all contour pixels
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version is '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    hull = cv2.convexHull(cnt)
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
    cv2.drawContours(drawing,[hull],0,(0,0,255),3)

## find convexity defects
    count_defects = fingernumber(cnt,drawing)

    if count_defects == 2 :
        cv2.putText(img,"forward!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 3:
        cv2.putText(img, "turn left!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(img,"turn right!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else :
        cv2.putText(img,"Stop!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    #cv2.imshow('drawing', drawing)
    #cv2.imshow('end', crop_img)
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)
##keyboard
    k = cv2.waitKey(10)
    if k == ord("q"):
        break
