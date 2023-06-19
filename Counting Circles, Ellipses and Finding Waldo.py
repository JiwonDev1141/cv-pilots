import cv2
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
import zipfile
from urllib.request import urlopen
# or: requests.get(url).content


def imshow(title="Image", image=None, size=12):
  w, h = image.shape[0], image.shape[1]
  aspect_ratio = w / h
  plt.figure(figsize=(size * aspect_ratio, size))
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  plt.title(title)
  plt.show()


resp = urlopen("https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip")
imagesZip = zipfile.ZipFile(BytesIO(resp.read()))

with zipfile.ZipFile(imagesZip, 'r') as zip_ref:
    zip_ref.extractall('images')