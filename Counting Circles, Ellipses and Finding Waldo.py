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

# 이미지 압축파일 다운로드 후 압축 해제
resp = urlopen("https://moderncomputervision.s3.eu-west-2.amazonaws.com/images.zip")

imagesZip = zipfile.ZipFile(BytesIO(resp.read())).extractall()

image = cv2.imread("images/blobs.jpg", 0)
imshow("Original Image", image)

# Intialize the detector using the default parameters
detector = cv2.SimpleBlobDetector_create()

# Detect blobs
keypoints = detector.detect(image)

# Draw blobs on our image as red circles
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Total Number of Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

# Display image with blob keypoints
imshow("Blobs using default parameters", blobs)

params = cv2.SimpleBlobDetector_Params()

# Set Circularity filtering parameters - blob이 얼마나 원에 가까운 지
params.filterByCircularity = True
params.minCircularity = 0.9

# 컨벡스 헐 알고리즘은 2차원 평면상에 여러개의 점이 있을 때 
# 그 점 중에서 일부를 이용하여 볼록 다각형을 만들되 볼록 다각형 내부에 모든 점을 포함시키는 것을 의미한다.
# Set Convexity filtering parameters - convexity: convex hull과 물체의 경계를 비교하여 가장 오목한 지점?
params.filterByConvexity = False
params.minConvexity = 0.2

# Set inertia filtering parameters - 원이면 1, 타원이면 0과 1 사이, 직선이면 0
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(image)

# Draw blobs on our image as red circles
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

imshow("Filtering Circular Blobs Only", blobs)


######### Where is Waldo #########

image = cv2.imread('./images/WaldoBeach.jpg')
imshow('Where is Waldo?', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

template = cv2.imread('./images/waldo.jpg', 0)

# 두개의 이미지를 비교해서 match되는 것 찾기
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Create Bounding Box
top_left = max_loc
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)

imshow("Where is Waldo", image)