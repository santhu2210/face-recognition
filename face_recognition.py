#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:35:46 2018

@author: root
"""

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

subjects = ["", "Brack Obama", "Validimir Putin", "Dr.Kalam" , "Breen"]

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print ("gray-->", gray)
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_alt.xml')
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=5);
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3);

    #print ("face found ----> ", len(faces))
    if (len(faces)==0):
        return None, None
    x, y , w, h = faces[0]
    return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        label = int(dir_name.replace("s",""))
       # print ("label- -> ", label)
        
        subject_dir_path = data_folder_path +"/"+dir_name
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            image_path = subject_dir_path+"/"+image_name
            #print ("image _ path:", image_path)
            image = cv2.imread(image_path)
           # plt.imshow(image)
            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    return faces, labels

print ("Preparing Data.....")
faces, labels = prepare_training_data("train_data")
print ("Data Prepared")

#print ("faces: ", faces)
print("Total faces: ", len(faces))
print("Total Labels: ", len(labels))

#print labels

#create our LBPH face recognizer
face_recongnizer = cv2.face.LBPHFaceRecognizer_create()

# create our LBPH face recongizer using fisher
#face_recongnizer = cv2.face.FisherFaceRecognizer_create()

#create eigen face reconginzer
#face_recongnizer = cv2.face.EigenFaceRecognizer_create()

face_recongnizer.train(faces, np.array(labels))

#function to draw rectangle on image
def draw_rectangle(img, rect):
    (x,y,w,h) = rect
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
# functio to draw in passed co-ordinates
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 2)
    
 
# convert to RGB
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# predict the subjects
def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    
    if face is not None:
                 
        #predict the imag using face_recongnizer
        label = face_recongnizer.predict(face)
        print ("preditceted label:", label , "confidenance at rank basis:",label[1] )
        label_text = subjects[label[0]]
        print ("label_text --> " , label_text , rect[0], rect[1])
        
        #draw the rect. around  face detected
        draw_rectangle(img, rect)
        
        #draw the text around predicted person
        draw_text(img, label_text, rect[0], rect[1]-4)
        return img
    else:
        print ("No face Detected ")
        return None        
 

print ("predicting images")

#load test images
test_img1 = cv2.imread("test_data/putin_test.jpg")
#test_img2 = cv2.imread("test_data/test2_img.jpg")

#performance prediction
predicted_img = predict(test_img1)
#predicted_img2 = predict(test_img2)

print("prediction completed")


# draw predicted image
if predicted_img is not None:
    plt.imshow(convertToRGB(predicted_img))
#cv2.imshow(subjects[1], predicted_img1)
#plt.imshow(predicted_img2)




            
            
            
        