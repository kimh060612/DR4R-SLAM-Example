import cv2
import numpy as np
from numpy.linalg import svd, det

K = np.array([
    [520.9, 0, 325.1],
    [0, 521.0, 249.7],
    [0, 0, 1]
])
min_dist = 987654321.0
max_dist = 0.0

def pixel2Cam(p: tuple):
    return (
        (p[0] - K[0][2]) / K[0][0],
        (p[1] - K[1][2]) / K[1][1]
    )

def ICP(pair1: list, pair2: list):
    N = len(pair1)
    center1 = np.array([0., 0., 0.])
    center2 = np.array([0., 0., 0.])
    for i in range(N):
        center1 += np.array(pair1[i])
        center2 += np.array(pair2[i])
    center1 /= N; center2 /= N
    
    q1 = []
    q2 = []
    for i in range(N):
        q1.append(np.array(pair1[i]) - center1)
        q2.append(np.array(pair2[i]) - center2)
    
    W = np.zeros((3, 3))
    for i in range(N):
        vec1 = np.expand_dims(q1[i], axis=1)
        vec2 = np.expand_dims(q2[i], axis=0)
        W += vec1 @ vec2
    U, _, Vt = svd(W) # Numpy SVD function
    R = U @ Vt
    if det(R) < 0:
        R = -R
    t = np.expand_dims(center1, axis=1) - (R @ np.expand_dims(center2, axis=1))
    return R, t
    

if __name__ == "__main__":

    imageA = cv2.imread('./data/1.png') # 오른쪽 사진
    imageB = cv2.imread('./data/2.png') # 왼쪽 사진
    imageADepth = cv2.imread('./data/1_depth.png', cv2.IMREAD_UNCHANGED) # 오른쪽 사진 RGB-D Depth Image
    imageBDepth = cv2.imread('./data/2_depth.png', cv2.IMREAD_UNCHANGED) # 왼쪽 사진 RGB-D Depth Image

    gray1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

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
    matches = matcher.match(desc1, desc2)

    points3DA = []
    points3DB = []

    # match distance between descriptors
    for i, con in enumerate(desc1):
        dist = matches[i].distance
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)

    for m in matches:
        if (m.distance > max(2 * min_dist, 30.)):
            # descriptor distance가 일정 수준 이상으로 커지면 outlier 처리
            continue
        x1, y1 = int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])
        x2, y2 = int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])
        d1, d2 = imageADepth[y1][x1], imageBDepth[y2][x2]
        if (d1 == 0) or (d2 == 0):
            continue
        p1 = pixel2Cam(kp1[m.queryIdx].pt)
        p2 = pixel2Cam(kp2[m.trainIdx].pt)
        dd1 = d1 / 5000.0; dd2 = d2 / 5000.0
        points3DA.append((p1[0] * dd1, p1[1] * dd1, dd1))
        points3DB.append((p2[0] * dd2, p2[1] * dd2, dd2))
    
    print("The number of matched point pairs:", len(points3DA))
    R, t = ICP(points3DA, points3DB)
    print("Rotation Matrix: ", R)
    print("Translation Vector: ", t)