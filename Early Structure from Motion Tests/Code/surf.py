import numpy as np 
from cv2 import cv2 

# Read the query image as query_img and traing image This query image  is what you need to find in train image 
# Save it in the same directory with the name image.jpg 

orginalquery_img = cv2.imread('top1.jpg') 
originaltrain_img = cv2.imread('top2.jpg') 

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
surf = cv2.xfeatures2d.SURF_create()

kp1, des1 = surf.detectAndCompute(query_img_bw,None)
kp2, des2 = surf.detectAndCompute(train_img_bw,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(query_img_bw,train_img_bw,k=2)

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        cv2.matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = cv2.matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

# draw the matches to the final image  containing both the images the drawMatches()  function takes both images and keypoints  and outputs the matched query image with its train image 
final_img = cv2.drawMatchesKnn(query_img,kp1,train_img,kp2,matches,None,**draw_params)

cv2.imshow("Matches", final_img) 
cv2.waitKey(0) 
