#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:06:46 2018

@author: shanthakumarp
"""

import cv2
import matplotlib.pyplot as plt

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_faces(f_cascade, colored_img, scaleFactor= 1.1):
    img_copy = colored_img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    print ("Total Faces: ", len(faces))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0, 255, 0), 4)
    
    return img_copy



# load cascade classifier for haarcascade
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# load test image
test1 = cv2.imread('sample_images/man_same_dog.jpg')

#call the detect function with image & classifier
face_detected_img = detect_faces(haar_face_cascade, test1, scaleFactor=1.1)

#convert the img to RGB and show
plt.imshow(convertToRGB(face_detected_img))

## copy the local image
#det_img = test1.copy()
#
##convert the test image to gray image for opencv
#gray_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
#
#plt.imshow(gray_img, cmap='gray')
#
## load cascade classifier for haarcascade
#haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#
#faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=8);
#
#print ('Face Found: ', len(faces))
#
#for (x,y,w,h) in faces:
#    cv2.rectangle(det_img, (x,y), (x+w, y+h), (0,255,0), 6)    
#    
#plt.imshow(convertToRGB(det_img))



