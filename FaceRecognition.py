# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:49:58 2019

@author: malve
"""

import numpy as np
import cv2

#cargamos la plantilla e inicializamos la webcam:
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while(True):
    #leemos un frame y lo guardamos
    ret, img = cap.read()
    
    #convertimos la imagen a blanco y negro
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    #Buscamos las coordenadas de los rostros (si los hay) y 
    #guardamos su posicion
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    profiles = profile_cascade.detectMultiScale(gray, 1.3,5)
    
    
    #Dibujamos un rectangulo en las coordenadas de cada rostro
    for(x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
               
        #cv2.ellipse(img, center, (w//2, h//2), 0, 0, 360, (11, 0, 64, 4))
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(125,255,0),2)
        faceROI = gray[y:y+h,x:x+w]
        eyes = eyes_cascade.detectMultiScale(faceROI)
        
        for(x2,y2,w2,h2) in eyes:
            center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2+h2)*0.25))
            img = cv2.ellipse(img, center, (w2//2, h2//2), 0, 0, 360, (255, 96, 255,3))
        
        #smiles = smile_cascade.detectMultiScale(faceROI)
        
        #for(x,y,w,h) in smiles:
        #    img = cv2.rectangle(img, (x,y),(x+w,y+h),(111,236,0),4)

    #cv2.imshow('img', img)
    
    for(x,y,w,h) in profiles:
        img = cv2.rectangle(img, (x,y),(x+w,y+h),(111,236,0),4)
    
        
    cv2.imshow('img', img)
    
    #con la tecla 'q' salimos del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    
cap.release()
cv2.destroyAllWindows()
