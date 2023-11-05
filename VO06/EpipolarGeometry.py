import numpy as np
import cv2

def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

K = np.array([[520.9, 0, 325.1], 
              [0, 521.0, 249.7], 
              [0, 0, 1]])

imageA = cv2.imread('./data/1.png') # 오른쪽 사진
imageB = cv2.imread('./data/2.png') # 왼쪽 사진

grayA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)

detector = cv2.ORB_create()
kp1, desc1 = detector.detectAndCompute(grayA, None)
kp2, desc2 = detector.detectAndCompute(grayB, None)

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
matches = matcher.knnMatch(desc1, desc2, k=2)
good = []
pts1 = []
pts2 = []

for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
E, mask = cv2.findEssentialMat(pts1,pts2,method=cv2.RANSAC,focal=521,maxIters=500,threshold=1)
points, R, t, mask = cv2.recoverPose(E, pts1, pts2)
print("Essential Matrix: ", E)
print("Rotation Matrix: ", R)
print("Translation Vector: ", t)

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(grayA, grayB, lines1, pts1, pts2)

lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(grayB, grayA, lines2, pts2, pts1)

cv2.imwrite('./data/line1.png', img5)
cv2.imwrite('./data/line2.png', img3)