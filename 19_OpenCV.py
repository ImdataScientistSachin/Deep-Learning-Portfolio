#!/usr/bin/env python
# coding: utf-8

# # Open CV (Computer Vision)

# #### OpenCV is a powerful library for computer vision that integrates seamlessly with deep learning frameworks to enable advanced image and video processing tasks.
# 
# ### Deep Learning Integration :
# ##### OpenCV supports popular deep learning frameworks such as TensorFlow, PyTorch, and Caffe. It provides tools for preprocessing data, deploying trained models, and optimizing inference performance
# 
# ###  Key features include
# ##### Pre-trained models for tasks like face detection and object segmentatin .
# ##### APIs for integrating deep learning models into computer vision pipelines..

# ###  Cascade Classifier

# ### VideoCapture


import cv2

"""
video = cv2.VideoCapture(0)
while True:
    _, frame = video.read()
    print(frame.shape)
    cv2.imshow("mycamera",frame)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
"""


# Initialize the video capture object using the default camera (index 0)
video = cv2.VideoCapture(0)

# Loop indefinitely until the user decides to exit
while True:
    # Read a frame from the video capture object
    # The variable '_' is used to ignore the return value (a boolean indicating success)
    # and only keep the frame
    _, frame = video.read()
    
    # Print the dimensions of the frame (height, width, channels)
    print(frame.shape)
    
    # Display the frame in a window titled "mycamera"
    cv2.imshow("mycamera", frame)
    
    # Wait for a key press for 10 milliseconds
    # This allows the window to update and respond to user input
    key = cv2.waitKey(10)
    
    # Check if the user pressed the 'q' key to exit the loop
    if key == ord('q'):
        break

# Release the video capture object to free up system resources
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


# ###  Cascade Classifier
# ### Front face Detecting

# ##### A basic face detection script using OpenCV's Haar cascade classifier. It captures video from the webcam, detects faces in each frame, and draws rectangles around them.


#testing cascad
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
# To capture video from webcam. 
# cap = cv2.VideoCapture(0)

# To use a video file as input 
cap = cv2.VideoCapture('testing.mp4')
# Check if the video capture has been initialized correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    
while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    #print(faces.shape)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 100, 255), 4)
        cv2.putText(img, str('sachin'), (x+5,y-5), font, 3, (255,0,0), 2)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()





# ###  Cascade Classifier

# ### Ip camera

# #### working


import requests
import cv2
import numpy as np
import imutils
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "http://192.168.68.189:8080/shot.jpg" 
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    cv2.imshow("Android_cam", img)
  
    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break
  
cv2.destroyAllWindows()
