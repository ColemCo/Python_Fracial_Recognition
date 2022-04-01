import numpy as np
import cv2
import pickle

#gets the cascade from opencv
facial_rec = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

#reads the numpy array file created by the image-rec.py
recognizer.read("trainer.yml")

#gets name from pickle file created in image-rec
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

#starts capture
cap = cv2.VideoCapture(0)

while(True):
    #captures frame-by-frame
    ret, frame = cap.read()

    #converts to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detects face and prints out the x,y,w,h location
    faces = facial_rec.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #brings in recognizer and prints ID and Name
        id_, conf = recognizer.predict(roi_gray)
        if conf >=45 and conf <=85:
            print(id_)
            print(labels[id_])

            #below prints name
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0,0,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        #colors and draws rectangle around the face
        color = (255,0,0)
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame,(x,y), (width, height), color, stroke)

    #displays the camera capture frame
    cv2.imshow('frame',frame)
    #quits capture at press of q
    if cv2.waitKey(40) == ord('q'):
        break

#closes capture window
cap.release()
cv2.destroyAllWindows()

