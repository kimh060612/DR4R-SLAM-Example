import numpy as np
import cv2
import matplotlib.pyplot as plt

UPPER_BOUND = 50
LOWER_BOUND = 10
TR_RANGE = UPPER_BOUND - LOWER_BOUND

def getColor(depth: np.ndarray):
    if (depth > UPPER_BOUND):
        depth = UPPER_BOUND;
    if (depth < LOWER_BOUND):
        depth = LOWER_BOUND;
    return int(255 * depth / TR_RANGE), int(0), int(255 * (1 - depth / TR_RANGE))

# Rescale to Homogeneous Coordinate
def rescale_point(pts1, pts2, length):
    a = [[]]
    b = [[]]
    for i in range(length):
        tmp1 = pts1[i].flatten()
        tmp1 = np.append(tmp1, 1)
        a = np.append(a, tmp1)
        tmp2 = pts2[i].flatten()
        tmp2 = np.append(tmp2, 1)
        b = np.append(b, tmp2)
    
    a = a.reshape((length),3)
    b = b.reshape((length),3)
    return a, b

# Triangulation
def LinearTriangulation(Rt0, Rt1, p1, p2):
    A = [p1[1]*Rt0[2,:] - Rt0[1,:], # x(p 3row) - (p 1row) 
        -(p1[0]*Rt0[2,:] - Rt0[0,:]), # y(p 3row) - (p 2row) 
        p2[1]*Rt1[2,:] - Rt1[1,:], # x'(p' 3row) - (p' 1row) 
        -(p2[0]*Rt1[2,:] - Rt1[0,:])]  # y'(p' 3row) - (p' 2row)
        
    A = np.array(A).reshape((4,4))
    AA = A.T @ A 
    U, S, VT = np.linalg.svd(AA) # right singular vector
 
    return VT[3,0:3]/VT[3,3]

img1 = cv2.imread('./data/1.png')
img2 = cv2.imread('./data/2.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

K = np.array([
    [520.9, 0.0, 325.1], 
    [0.0, 521.0, 249.7], 
    [0.0, 0.0, 1.0]
])

detector = cv2.ORB_create()
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

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
E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, focal=521, maxIters=500, threshold=1)
points, R, t, mask, triP = cv2.recoverPose(E, pts1, pts2, K, distanceThresh=0.5)
depth = triP[2, :]
color = list(map(getColor, depth))

# Generate 3D point by implementing Triangulation
Rt0 = np.hstack((np.eye(3), np.zeros((3, 1))))
Rt1 = np.hstack((R, t))
Rt1 = np.matmul(K, Rt1)

pt1 = np.transpose(pts1)
pt2 = np.transpose(pts2)

p1, p2 = rescale_point(pts1, pts2, len(pts1))
p3ds = []
for pt1, pt2 in zip(p1, p2):
    p3d = LinearTriangulation(Rt0, Rt1, pt1, pt2)
    p3ds.append(p3d)
p3ds = np.array(p3ds).T

X = np.array([])
Y = np.array([])
Z = np.array([]) #120 
X = np.concatenate((X, p3ds[0]))
Y = np.concatenate((Y, p3ds[1]))
Z = np.concatenate((Z, p3ds[2]))

fig = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c='b', marker='o') 
plt.savefig('./data/triangulation.png')

retImg1 = img1[:,:,:]
retImg2 = img2[:,:,:]
for i in range(len(color)):
    h1, w1 = pts1[i]
    h2, w2 = pts2[i]
    cv2.circle(retImg1, (int(h1), int(w1)), 2, color[i], 2);
    cv2.circle(retImg2, (int(h2), int(w2)), 2, color[i], 2);

cv2.imwrite('./data/triangulation1.png', retImg1)
cv2.imwrite('./data/triangulation2.png', retImg2)
