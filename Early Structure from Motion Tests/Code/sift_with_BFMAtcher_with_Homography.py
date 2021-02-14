import numpy as np 
from cv2 import cv2 
import matplotlib.pyplot as plt
# Read the query image as query_img and traing image This query image  is what you need to find in train image 
# Save it in the same directory with the name image.jpg 

orginalquery_img = cv2.imread('img1.jpg') 
originaltrain_img = cv2.imread('img2.jpg') 

scale_percent = 30  # percent of original size
width1 = int(orginalquery_img.shape[1] * scale_percent / 100)
height1 = int(orginalquery_img.shape[0] * scale_percent / 100)
dim = (width1, height1)
query_img = cv2.resize(orginalquery_img, dim, interpolation=cv2.INTER_AREA)

scale_percent = 30  # percent of original size
width2 = int(originaltrain_img.shape[1] * scale_percent / 100)
height2 = int(originaltrain_img.shape[0] * scale_percent / 100)
dim = (width2, height2)
train_img = cv2.resize(originaltrain_img, dim, interpolation=cv2.INTER_AREA)

# Convert it to grayscale 
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY) 
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 

# Initialize the ORB detector algorithm 
sift = cv2.SIFT_create(400)

# Now detect the keypoints and compute  the descriptors for the query image and train image 
kp1, des1 = sift.detectAndCompute(query_img_bw,None)
kp2, des2 = sift.detectAndCompute(train_img_bw,None)

# Initialize the Matcher for matching  the keypoints and then match the  keypoints 
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# draw the matches to the final image  containing both the images the drawMatches()  function takes both images and keypoints  and outputs the matched query image with its train image 
final_img = cv2.drawMatchesKnn(query_img,kp1,train_img,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("Matches", final_img) 
cv2.waitKey(0) 

#############################################
#                                           #
#                                           #
#                                           #
#############################################

points1 = np.zeros((len(matches), 2), dtype=np.float32)  #Prints empty array of size equal to (matches, 2)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
   points1[i, :] = kp1[match[0].queryIdx].pt    #gives index of the descriptor in the list of query descriptors
   points2[i, :] = kp2[match[0].trainIdx].pt    #gives index of the descriptor in the list of train descriptors

print(points1)
print(points1[0,0])
print(points1[0,1])

x = des1[0][0]
y = des1[0][1]

#Now we have all good keypoints so we are ready for homography.   
# Find homography
  
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
height, width, channels = train_img.shape
im1Reg = cv2.warpPerspective(query_img, h, (width, height))  #Applies a perspective transformation to an image.
print("Estimated homography : \n",  h)

#for i, match in enumerate(matches):

p  = [x, y, 1]
projection = h * p
projnorm   = projection / [0,0,1]

print(projnorm[0][2])
print(projnorm[1][2])
print(projnorm[2][2])

x1 = projnorm[0][2]
y1 = projnorm[1][2]
z1 = projnorm[2][2]

cv2.imshow("Registered image", im1Reg)
cv2.waitKey(0)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x1, y1, z1, cmap='viridis', linewidth=0.5)
plt.show()

