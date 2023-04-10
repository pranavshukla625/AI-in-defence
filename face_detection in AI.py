from PIL.Image import Image
import face_recognition as fr
import cv2 
import numpy as np
import os
from datetime import datetime
path = 'IMG' #Your Folder is to be mention here 
images = []
classNames =[]
myList = os.listdir(path)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def marking(Name):
    with open('logs.csv', 'r+') as f:
        data = f.readlines()
        namelist = []
        for line in data:
            entry = line.split(',')
            namelist.append(entry[0])
            if name not in namelist:
                now = datetime.now()
                DtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{DtString}')

encodelistknown = findEncodings(images)
print("Encoding Completed")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, int(60))

while True: 
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = fr.face_locations(imgS)
    encodesCurFrame = fr.face_encodings(imgS, facesCurFrame)

    for encodeface, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = fr.compare_faces(encodelistknown, encodeface)
        faceDis = fr.face_distance(encodelistknown, encodeface)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),1)
            marking('name')

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
