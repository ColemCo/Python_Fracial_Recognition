import os
import cv2
import numpy as np
from PIL import Image
import pickle

#establishes the base directory where the image folder is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

#gets the cascade from opencv
facial_rec = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_id = {}
y_labels = []
x_train = []

#finds and identifies any pictures in the images folder
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            # is the path of all the pictures in the images folder
            path = os.path.join(root, file)
            # label is the name of the folder, which is the persons name
            label = os.path.basename(root)
            #print(label, path)

            # uses numbers to give a label to each person
            if not label in label_id:
                label_id[label] = current_id
                current_id += 1
            id_ = label_id[label]
            #print(label_id)

            # makes the pictures into gray scale
            pil_image = Image.open(path).convert("L")

            # converts each image to a number array using numpy
            image_array = np.array(pil_image, "uint8")
            #print(image_array)

            # runs the image array through opencv
            faces= facial_rec.detectMultiScale(image_array)

            # sets the reagion of interest for the images and connects roi with x_train
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+h]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)

#creates file with all of the labels
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_id, f)

#trains the recognizer with the numpy arrays
recognizer.train(x_train, np.array(y_labels))
#creates a file of the numpy array to use transfer to main code
recognizer.save("trainer.yml")