import cv2
import numpy as np
from matplotlib import pyplot as plt
from urllib.request import urlopen
import zipfile
from io import BytesIO


resp = urlopen("https://moderncomputervision.s3.eu-west-2.amazonaws.com/haarcascades.zip")

haarcascadesZip = zipfile.ZipFile(BytesIO(resp.read())).extractall()

# Define our imshow function
def imshow(title = "image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

# classifier: 분류기
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

image = cv2.imread('images/Trump.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)

if faces is ():
    print("No faces found")

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (127,0,255), 2)

imshow('Face Detection', image)

######################################3

################# Simple Eye & Face Detection using Haarcascade Classifiers 

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

img = cv2.imread('images/Trump.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

if faces is ():
    print("No Face Found")

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (127,0,255), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes =  eye_classifier.detectMultiScale(roi_gray, 1.2, 3)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,255,0), 2)

imshow('Eye & Face Detection', img)

####################################

############# Using Colab's Code Snippets let's access the webcam for an input

from Ipython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

             // Resize the output to fit the video element.
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d).drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            
            return canvas.toDataURL('image/jpeg', quality);

        }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])

    with open(filename, 'wb') as f:
        f.write(binary)
    return filename