
# coding: utf-8

# In[21]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import numpy.linalg as linalg
import cv2
import random
from mpl_toolkits.mplot3d import Axes3D
#get_ipython().magic(u'matplotlib notebook')
orb = cv2.ORB_create()
N = 200
s = 8
random.seed(0xDEADBEEF)


# In[22]:


#function for computing the fundamental matrix
def computeF(xl, xr):
    A = []
    for i in range(len(xl)):
        x = xl[i]
        x_prime = xr[i]
        #Put a row into the matrix, with formula fropm class.
        A.append([x[0]*x_prime[0], x[1]*x_prime[0], x_prime[0],
                  x[0]*x_prime[1], x[1]*x_prime[1], x_prime[1], x[0], x[1], 1])
    #Get actual fundamental matrix.
    A = np.array(A)
    _, _, V = linalg.svd(A)
    F = V[-1,:].reshape((3,3))
    U, s, V = linalg.svd(F)
    sigma = np.diag(s)
    sigma[2,2] = 0
    return np.matmul(U, np.matmul(sigma, V))
    
def ransacF(points, kp1, kp2):
    n = 0 
    currF = None
    maxInliers = 0
    # run ransac
    for n in range(N):
        sample = random.sample(points, s)
        leftPoints = [kp1[point.trainIdx].pt for point in sample]
        rightPoints = [kp2[point.queryIdx].pt for point in sample]
        # compute our F matrix
        FToTest = computeF(leftPoints, rightPoints)
        numIn = 0
        for match in points:
            p = kp1[match.trainIdx].pt
            q = kp2[match.queryIdx].pt
            if (abs(np.matmul([p[0], p[1], 1],
                              np.matmul(FToTest, [q[0], q[1], 1])))) < .1:
                numIn += 1
        if currF is None:
            currF = FToTest
        if numIn > maxInliers:
            maxInliers = numIn
            currF = FToTest
    return currF


# In[23]:


npz = np.load("calibration/matrices.npz")
K = npz['cameraMatrix']
K_prime = np.transpose(K)
W = np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
negW = np.transpose(W)
Z = np.matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
trans = np.transpose
svd = linalg.svd
matmul = np.matmul
location = np.reshape((np.array([0, 0, 0, 1])), (4, 1))
# Compute null space of matrix A, algorithm from scipy mailing lists
def null(A, eps=1e-15):
    u, s, vh = linalg.svd(A)
    null_mask = (s <= eps)
    null_space = np.compress(null_mask, vh, axis=0)
    return trans(null_space)

def findPointsFront(points, R, t):
    # TODO find number of points in front of both cameras
    return 0

def updateLoc(F, points):
    global location
    # TODO
    # Need to find K for the camera
    E = matmul(K, matmul(F, K_prime))
    # https://en.wikipedia.org/wiki/Essential_matrix
    U, sigma, Vt = svd(E)
    t_mat = matmul(U, matmul(Z, trans(U)))
    RNeg = matmul(U, matmul(negW, Vt))
    RPos = matmul(U, matmul(W, Vt))
    opt = 0
    t = null(t_mat)
    neg_t = -1 * t
    # There are four valid comboes, we need to find the one with points in front of image
    # can probably parallelize this? should profile to see if worth
    numPoints1 = findPointsFront(points, RNeg, neg_t)
    numPoints2 = findPointsFront(points, RNeg, t)
    numPoints3 = findPointsFront(points, RPos, neg_t)
    numPoints4 = findPointsFront(points, RPos, t)
    whichNum = np.argmax([numPoints1, numPoints2, numPoints3, numPoints4])
    whichNum = 0
    # currently grabs the first option by default
    if whichNum == 0:
        motion = np.hstack([RNeg, neg_t])
        motion = np.vstack([motion, np.array([0, 0, 0, 1])])
    elif whichNum == 1: 
        motion = np.hstack(RNeg, t)
        motion = np.vstack([motion, np.array([0, 0, 0, 1])])
    elif whichNum == 2:
        motion = np.hstack([RPos, neg_t])
        motion = np.vstack([motion, np.array([0, 0, 0, 1])])
    else:
        motion = np.hstack([RPos, t])
        motion = np.vstack([motion, np.array([0, 0, 0, 1])])
    location = np.reshape((np.matmul(motion, location)), (4, 1))
    return None


# In[24]:


cap = cv2.VideoCapture('motion.mp4')
i = 0
last = None
lastDes = None
locations = []
now = time.time()
while cap.isOpened():
    i += 1
    
    ret, frame = cap.read()
    if i % 10 != 0:
        continue
    if ret == False:
        break
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(grayImg, None)
    
    if last == None:
        last = kp
        lastDes = des
        continue
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(lastDes, des, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    F = ransacF(good, kp, last)
    updateLoc(F, good)
    locations.append(location)
    last = kp
    lastDes = des
    
after = time.time()
print("time elapsed: {0}".format(after - now))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot([np.reshape(location, (4,1))[0,0] for location in locations],
       [np.reshape(location, (4,1))[1,0] for location in locations],
       [np.reshape(location, (4,1))[2,0] for location in locations],
       label="locations")
plt.show()


# In[ ]:




