#!/usr/bin/env python3
# coding: utf-8

# System Imports
import random
import time
from multiprocessing import Queue, Process
import multiprocessing as mp

# 3rd Party
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as linalg


# Initialization
orb = cv2.ORB_create()
N = 150
s = 8
random.seed(0xDEADBEEF)
# Get the calibration matrix
npz = np.load("calibration/matrices.npz")
K = npz['cameraMatrix']
K_prime = np.transpose(K)
Kinv = linalg.inv(K)

W = np.matrix( [[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]] )
Winv = np.matrix( [[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]] )

negW = np.transpose(W)
Z = np.matrix( [[0, 1, 0],
                [-1, 0, 0],
                [0, 0, 0]] )
trans = np.transpose
svd = linalg.svd
matmul = np.matmul
location = np.reshape( (np.array([0, 0, 0, 1])), (4, 1) )


def computeF(xl, xr):
    """Calculate a fundamental matrix based on
    the corresponding features from two images.

    Arguments:
    xl:list - features from the left image.
    xr:list - features from the right image.

    Returns:
    The approximate fundamental matrix that best
    fits the given correspondences.
    """
    A = []
    for i in range(len(xl)):
        x = xl[i]
        x_prime = xr[i]
        #Put a row into the matrix, with formula from class.
        A.append([ x[0]*x_prime[0], x[1]*x_prime[0], x_prime[0],
                   x[0]*x_prime[1], x[1]*x_prime[1], x_prime[1], x[0], x[1], 1
                 ])
    #Get actual fundamental matrix.
    A = np.array(A)
    _, _, V = linalg.svd(A)
    F = V[-1,:].reshape((3,3))
    U, s, V = linalg.svd(F)
    sigma = np.diag(s)
    # Create sigma prime.
    sigma[2,2] = 0
    return np.matmul(U, np.matmul(sigma, V))

#Performs RANSAC to estimate the fundamental matrix for the matches.
def ransacF(points, kp1, kp2):
    """Calculate the fundamental matrix for given keypoints,
    and check if the model is valid.

    Tries to maximize the number of inliers for a given
    matrix by using RANSAC.

    Arguments:
    points:list - The correspondences
    kp1:list - keypoints from the left image
    kp2:list - keypoints from the second image

    Returns:
    The best fitting fundamental matrix
    """

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


def null(A, eps=1e-15):
    """Compute null space of matrix A, algorithm from scipy mailing lists"""
    u, s, vh = linalg.svd(A)
    null_mask = (s <= eps)
    null_space = np.compress(null_mask, vh, axis=0)
    return trans(null_space)


def triangulate(F, kp1, kp2):
    """Calculate the essential matrix, and decompose it to create
    the projection matrix.

    Returns two projection matrices
    """
    E = matmul(K_prime, matmul(F, K))
    # https://en.wikipedia.org/wiki/Essential_matrix
    U, sigma, Vt = svd(E)
    t_mat = matmul(U, matmul(Z, trans(U)))
    #t_mat = U[:,2]
    RPos = matmul(U, matmul(W, Vt))
    opt = 0
    t_null = null(t_mat)
    neg_t = -1 * t_null

    proj1 = np.hstack([np.eye(3), np.matrix([[0.],[0.],[0.]])] )
    proj2 = np.hstack([RPos, t_mat])

    locations = []
    for p1, p2 in zip(kp1, kp2):
        p1_x = p1.pt[0]
        p1_y = p1.pt[1]
        p2_x = p2.pt[0]
        p2_y = p2.pt[1]
        p1_hom = matmul( Kinv, np.matrix( [[p1_x], [p1_y], [1]] ) )
        p2_hom = matmul( Kinv, np.matrix( [[p2_x], [p2_y], [1] ] ) )

        x = linearLSTriangulation(p1_hom.flatten(), proj1,
                                  p2_hom.flatten(), proj2)
        locations.append( np.matrix(x) )
    #location = np.reshape((np.matmul(motion, location)), (4, 1))
    return locations


def linearLSTriangulation(p1, proj1, p2, proj2):
    A = np.matrix([
            [ p1[0,0]*proj1[2,0]-proj1[0,0], p1[0,0]*proj1[2,1]-proj1[0,1], p1[0,0]*proj1[2,2]-proj1[0,2] ],
            [ p1[0,1]*proj1[2,0]-proj1[1,0], p1[0,1]*proj1[2,1]-proj1[1,1], p1[0,1]*proj1[2,2]-proj1[1,2]],
            [ p2[0,0]*proj2[2,0]-proj2[0,0], p2[0,0]*proj2[2,1]-proj2[0,1], p2[0,0]*proj2[2,2]-proj2[0,2]],
            [ p2[0,1]*proj2[2,0]-proj2[1,0], p2[0,1]*proj2[2,1]-proj2[1,1], p2[0,1]*proj2[2,2]-proj2[1,2]],
        ])
    b = np.matrix([
        [ p1[0,0]*proj1[2,3] - proj1[0,3] ],
        [ p1[0,1]*proj1[2,3] - proj1[1,3] ],
        [ p2[0,0]*proj2[2,3] - proj2[0,3] ],
        [ p2[0,1]*proj2[2,3] - proj2[1,3] ]
        ])
    #print(A.shape)
    #print(b.shape)
    x = np.zeros( (3,1) )
    _, x = cv2.solve(A, b, x, cv2.DECOMP_SVD)
    return x


def queueFrames(q):
    cap = cv2.VideoCapture('motion.mp4')
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            q.put(None)
            return
        if i % 10 == 0:
            grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            q.put(grayImg)
        i += 1


if __name__ == '__main__':
    last = None
    lastDes = None
    import sys
    if sys.version_info >= (3, 0):
        mp.set_start_method("spawn")
    q = Queue()
    locations = []
    now = time.time()
    #start reading image frames in background
    Process(target=queueFrames, args=(q,)).start()
    while True:
        grayImg = q.get(True, timeout=100)
        if grayImg is None:
            break
        kp, des = orb.detectAndCompute(grayImg, None)

        # Initial cycle, so can't process further.
        if last == None:
            last = kp
            lastDes = des
            continue
        # BFMatcher with hamming distance because we're using ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(lastDes, des)
        F = ransacF(matches, kp, last)
        #location = triangulate(F, kp, last)
        # Create an array of matched keypoints for both images
        left = [ kp[m.trainIdx] for m in matches ]
        right = [ last[m.queryIdx] for m in matches ]
        locations += triangulate(F, left, right)
        last = kp
        lastDes = des

    after = time.time()
    print("time elapsed: {0}".format(after - now))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Matches won't print
    ax.plot([np.reshape(location, (3,1))[0,0] for location in locations],
           [np.reshape(location, (3,1))[1,0] for location in locations],
           [np.reshape(location, (3,1))[2,0] for location in locations],
           "ro", label="locations")
    plt.show()
