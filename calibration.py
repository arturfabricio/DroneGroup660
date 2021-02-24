import numpy as np
from cv2 import cv2
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

## Global Variables ##
mtx = []
dist = []
rvecs = []
tvecs = []
h = []

## Images for testing ##
orginalquery_img = cv2.imread('1_1.jpg') 

originaltrain_img = cv2.imread('2_2.jpg')

def cameraCalibration():
    print("cameraCalibration")
    global mtx, dist, rvecs, tvecs

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('/home/artur/Desktop/Calibration/final_checkerboard/*.jpg')
    for fname in images:
        img = cv2.imread(fname)

        # scale_percent = 20  # percent of original size
        # width1 = int(img.shape[1] * scale_percent / 100)
        # height1 = int(img.shape[0] * scale_percent / 100)
        # dim = (width1, height1)
        # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6,9), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(0)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Reprojection Error: ", ret)
    print("Camera Matrix: ", mtx)
    print("Distortion Coefficients: ", dist)

    img = cv2.imread('/home/artur/Desktop/Calibration/OutsidePics/1.jpg')
    
    # scale_percent = 20  # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Original', img)
    cv2.waitKey(0)

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    #crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imshow('Undistorted', dst)
    cv2.waitKey(0)

def findFeatures():
    print("findFeatures")
    global orginalquery_img
    global originaltrain_img
    global train_img
    global query_img
    global mtx
    global dist

    #orginalquery_img = cv2.undistort(orginalquery_img, mtx, dist, None, mtx)
    #originaltrain_img = cv2.undistort(originaltrain_img, mtx, dist, None, mtx)

    scale_percent = 20  # percent of original size
    width1 = int(orginalquery_img.shape[1] * scale_percent / 100)
    height1 = int(orginalquery_img.shape[0] * scale_percent / 100)
    dim = (width1, height1)
    query_img = cv2.resize(orginalquery_img, dim, interpolation=cv2.INTER_AREA)
    
    ##Undistort
    query_img = cv2.undistort(query_img, mtx, dist, None, mtx)
    cv2.imshow("Query_Img_Undistorted", query_img) 
    cv2.waitKey(0)

    scale_percent = 20  # percent of original size
    width2 = int(originaltrain_img.shape[1] * scale_percent / 100)
    height2 = int(originaltrain_img.shape[0] * scale_percent / 100)
    dim = (width2, height2)
    train_img = cv2.resize(originaltrain_img, dim, interpolation=cv2.INTER_AREA)
    
    ##Undistort
    train_img = cv2.undistort(train_img, mtx, dist, None, mtx)
    cv2.imshow("Train_Img_Undistorted", train_img) 
    cv2.waitKey(0) 

    # Convert it to grayscale 
    query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY) 
    train_img_bw = cv2.cvtColor(train_img,cv2.COLOR_BGR2GRAY) 

    # Initialize the ORB detector algorithm 
    sift = cv2.SIFT_create(700)

    # Now detect the keypoints and compute  the descriptors for the query image and train image 
    kp1, des1 = sift.detectAndCompute(query_img_bw,None)
    kp2, des2 = sift.detectAndCompute(train_img_bw,None)

    print("Points detected: ",len(kp1), " and ", len(kp2))

    # Initialize the Matcher for matching  the keypoints and then match the  keypoints 
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.70*n.distance:
            good.append([m])

    # draw the matches to the final image  containing both the images the drawMatches()  function takes both images and keypoints  and outputs the matched query image with its train image 
    final_img = cv2.drawMatchesKnn(query_img,kp1,train_img,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("Matches", final_img) 
    cv2.waitKey(0) 

    points1 = np.zeros((len(matches), 2), dtype=np.float)  #Prints empty array of size equal to (matches, 2)
    points2 = np.zeros((len(matches), 2), dtype=np.float)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match[0].queryIdx].pt    #gives index of the descriptor in the list of query descriptors
        points2[i, :] = kp2[match[0].trainIdx].pt    #gives index of the descriptor in the list of train descriptors

    print(np.shape(points1))

    #Now we have all good keypoints so we are ready for homography.   
    # Find homography
    

    ###############
    points1 = np.int32(points1)
    points2 = np.int32(points2)
    F, mask = cv2.findFundamentalMat(points1,points2,cv2.FM_LMEDS)
    print(F)
    #########

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width, channels = train_img.shape
    im1Reg = cv2.warpPerspective(query_img, h, (width, height))  #Applies a perspective transformation to an image.
    print("Estimated homography : \n",  h)

    cv2.imshow("Homography", im1Reg) 
    cv2.waitKey(0) 





    # k = mtx

    # R0 = np.matrix('1 0 0; 0 1 0; 0 0 1')
    # T0 = np.matrix('0;0;0') 

    # P1 = np.concatenate((np.dot(k,R0),np.dot(k,T0)), axis = 1)
    # print('Projection matrix 1: ', P1)
    # print(np.shape(P1))

    # num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(h, k)
    # print('Rotation matrix: ' , Rs)
    # print('Translation vector: ' , Ts)
    # P2 = np.concatenate((np.dot(k,Rs[1]),np.dot(k,Ts[1])), axis = 1)
    # print('Projection matrix 2: ', P2)

    # pts4D = cv2.triangulatePoints(P1,P2,points1.T,points2.T).T

    # ########################
    # ### UNDISTORT POINTS ###
    # ########################

    # pts3D = pts4D[:, :3]/np.repeat(pts4D[:, 3], 3).reshape(-1, 3)
    
    # # plot with matplotlib
    # Ys = pts3D[:, 0]
    # Zs = pts3D[:, 1]
    # Xs = pts3D[:, 2]

    # #Tpt1 = np.array([Tpoints1[0],Tpoints1[1],Tpoints1[2]])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(Xs, Ys, Zs, c='r', marker='o')
    # ax.set_xlabel('Y')
    # ax.set_ylabel('Z')
    # ax.set_zlabel('X')
    # plt.title('3D point cloud: Use pan axes button below to inspect')
    # plt.show()

cameraCalibration()
#findFeatures()


