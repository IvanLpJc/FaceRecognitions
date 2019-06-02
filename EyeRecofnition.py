# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:43:45 2019

@author: malve
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:49:58 2019

@author: malve
"""

import numpy as np
import cv2


#cargamos la plantilla e inicializamos la webcam:
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
cap = cv2.VideoCapture(0)

print("I've finished to train")

while(True):
    #leemos un frame y lo guardamos
    ret, img = cap.read()
   
    #convertimos la imagen a blanco y negro
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    #Buscamos las coordenadas de los rostros (si los hay) y 
    #guardamos su posicion
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    
    #Dibujamos un rectangulo en las coordenadas de cada rostro
    for(x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
               
        #cv2.ellipse(img, center, (w//2, h//2), 0, 0, 360, (11, 0, 64, 4))
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(125,255,0),2)
        faceROI = gray[y:y+h,x:x+w]
        smiles = smile_cascade.detectMultiScale(faceROI)
        
        for(x,y,w,h) in smiles:
            img = cv2.rectangle(img, (x,y),(x+w,y+h),(111,236,0),4)
        
    #Mostramos la imagen
    cv2.imshow('img', img)
    
    #con la tecla 'q' salimos del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
