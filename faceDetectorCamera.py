'''
face detector simple script.
which detects all faces in the background.

+=======================================+
| By       : Majid Isss                 |
| Email    : majidissa82@outlook.com    |
| WhatsApp : 01017398758                |
+=======================================+
'''

import tkinter as tk
from PIL import Image ,ImageTk
import cv2
import numpy as np

cap = cv2.VideoCapture(1)

# import classify the haar cascade face detector file.
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def run():
	_ ,frame = cap.read()
	frame = cv2.resize(frame ,(450,550))
	frame = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB)
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	
	face_poins = face_classifier.detectMultiScale(image=gray_frame ,scaleFactor=1.05 ,minNeighbors=5 ,minSize=(30,30))
	for x,y,w,h in face_poins:
		draw(x,y,w,h,20,frame)
		#cv2.rectangle(frame ,(x,y) ,(x+w,y+h) ,(0,255,0) ,2)
	
	img = ImageTk.PhotoImage(Image.fromarray(frame))
	
	video_label.imgtk = img
	video_label.configure(image=img)
	video_label.after(5,run)
	NO_Faces = len(face_poins)
	no_faces_label.configure(text='NO.Faces: {}'.format(NO_Faces))

# the method that draw around the face
def draw(x,y,w,h,l,img):
	pts1 = np.array([[x+l,y] ,[x,y] ,[x,y+l]])
	pts2 = np.array([[x+w-l,y] ,[x+w,y] ,[x+w,y+l]])
	pts3 = np.array([[x+l,y+h] ,[x,y+h] ,[x,y+h-l]])
	pts4 = np.array([[x+w-l,y+h] ,[x+w,y+h] ,[x+w,y+h-l]])
	
	# draw poly lines around the face
	cv2.polylines(img ,[pts1,pts2,pts3,pts4] ,False ,(0,255,100) ,2)
	

root = tk.Tk()

video_label = tk.Label(root)
video_label.pack()

no_faces_label = tk.Label(root)
no_faces_label.pack()

run()
root.mainloop()
