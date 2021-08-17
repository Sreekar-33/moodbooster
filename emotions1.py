from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from pygame import mixer
from tkinter import *
import os
import pygame
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("rate", 150)
engine.setProperty("voice", voices[1].id)

def playsong():
    currentsong = playlist.get(ACTIVE)
    mixer.music.load(currentsong)
    songstatus.set("Playing")
    mixer.music.play()


def pausesong():
    songstatus.set("Paused")
    mixer.music.pause()


def stopsong():
    songstatus.set("Stopped")
    mixer.music.stop()


def resumesong():
    songstatus.set("Resuming")
    mixer.music.unpause()



face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    text=""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Emotion Detector', frame)
            if label == 'Sad' or label == 'Angry' or label=='Happy':
                index = 0
                root = Tk()
                root.title('Mood Booster')

                mixer.init()
                songstatus = StringVar()
                songstatus.set("choosing")

                # playlist---------------

                playlist = Listbox(root, selectmode=SINGLE, bg="black", fg="white", font=('arial', 15), width=40)
                playlist.grid(columnspan=5)

                if label=='Sad':
                    os.chdir(r'C:\Users\saich\Desktop\face detection updated\songs\sad')
                    songs = os.listdir()
                    text = "Hey you look soo sad let me play a song for boosting up your mood. keep smiling"
                elif label == 'Angry':
                    os.chdir(r'C:\Users\saich\Desktop\face detection updated\songs\angry')
                    songs = os.listdir()
                    text = "your angry let me play you a song"
                elif label=='Happy':
                    os.chdir(r'C:\Users\saich\Desktop\face detection updated\songs\happy')
                    songs = os.listdir()
                    text = "Your Happy Lets vibe together"
                for s in songs:
                    playlist.insert(END, s)
                engine.say(text)
                engine.runAndWait()
                playsong()

                playbtn = Button(root, text="Play", command=playsong)
                playbtn.config(font=('arial', 20), bg="black", fg="white", padx=7, pady=7)
                playbtn.grid(row=1, column=0)

                pausebtn = Button(root, text="Pause", command=pausesong)
                pausebtn.config(font=('arial', 20), bg="black", fg="white", padx=7, pady=7)
                pausebtn.grid(row=1, column=1)

                stopbtn = Button(root, text="Stop", command=stopsong)
                stopbtn.config(font=('arial', 20), bg="black", fg="white", padx=7, pady=7)
                stopbtn.grid(row=1, column=2)

                Resumebtn = Button(root, text="Resume", command=resumesong)
                Resumebtn.config(font=('arial', 20), bg="black", fg="white", padx=7, pady=7)
                Resumebtn.grid(row=1, column=3)


                root.after(10000, root.destroy)
                root.mainloop()
                mixer.music.stop()


        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()