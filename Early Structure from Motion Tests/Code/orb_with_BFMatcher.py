import numpy as np 
from cv2 import cv2 

# Read the query image as query_img and traing image This query image  is what you need to find in train image 
# Save it in the same directory with the name image.jpg 

orginalquery_img = cv2.imread('img1.jpg') 
originaltrain_img = cv2.imread('img2.jpg') 

scale_percent = 40  # percent of original size
width1 = int(orginalquery_img.shape[1] * scale_percent / 100)
height1 = int(orginalquery_img.shape[0] * scale_percent / 100)
dim = (width1, height1)
query_img = cv2.resize(orginalquery_img, dim, interpolation=cv2.INTER_AREA)

scale_percent = 40  # percent of original size
width2 = int(originaltrain_img.shape[1] * scale_percent / 100)
height2 = int(originaltrain_img.shape[0] * scale_percent / 100)
dim = (width2, height2)
train_img = cv2.resize(originaltrain_img, dim, interpolation=cv2.INTER_AREA)

# Convert it to grayscale 
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY) 
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 

# Initialize the ORB detector algorithm 
orb = cv2.ORB_create() 

# Now detect the keypoints and compute  the descriptors for the query image and train image 
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None) 
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None) 

# Initialize the Matcher for matching  the keypoints and then match the  keypoints 
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
matches = matcher.match(queryDescriptors,trainDescriptors) 

# draw the matches to the final image  containing both the images the drawMatches()  function takes both images and keypoints  and outputs the matched query image with its train image 
final_img = cv2.drawMatches(query_img, queryKeypoints, 
train_img, trainKeypoints, matches[:300],None) 

# Show the final image 
cv2.imshow("Matches", final_img) 
cv2.waitKey(0) 
