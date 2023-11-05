import cv2
import numpy as np

K = np.array([
    [520.9, 0, 325.1],
    [0, 521.0, 249.7],
    [0, 0, 1]
])
distortions = np.zeros((4, 1))
min_dist = 987654321.0
max_dist = 0.0

UPPER_BOUND = 50
LOWER_BOUND = 10
TR_RANGE = UPPER_BOUND - LOWER_BOUND

def getColor(depth: np.ndarray):
    if (depth > UPPER_BOUND):
        depth = UPPER_BOUND;
    if (depth < LOWER_BOUND):
        depth = LOWER_BOUND;
    return int(255 * depth / TR_RANGE), int(0), int(255 * (1 - depth / TR_RANGE))

def pixel2Cam(p: tuple):
    return [
        (p[0] - K[0][2]) / K[0][0],
        (p[1] - K[1][2]) / K[1][1]
    ]


if __name__ == "__main__":

    imageA = cv2.imread('./data/1.png') # 오른쪽 사진
    imageB = cv2.imread('./data/2.png') # 왼쪽 사진

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

    pointsA = []
    pointsB = []
    pointCam1 = []
    pointCam2 = []

    for i, con in enumerate(desc1):
        dist = matches[i].distance
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)

    for m in matches:
        if (m.distance > max(2 * min_dist, 30.)):
            continue
        pointsA.append(kp1[m.queryIdx].pt)
        pointsB.append(kp2[m.trainIdx].pt)
        pointCam1.append(pixel2Cam(kp1[m.queryIdx].pt))
        pointCam2.append(pixel2Cam(kp2[m.trainIdx].pt))
    
    pts1 = np.int32(pointsA)
    pts2 = np.int32(pointsB)
    E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, focal=521, maxIters=500, threshold=1) # Essential Matrix 계산
    points, R, t, mask, triP = cv2.recoverPose(E, pts1, pts2, K, distanceThresh=0.5) # Essential Matrix로부터 Camera pose 계산
    T1 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    T2 = np.concatenate([R, t], axis=1)
    print(T1, T2, sep="\n")
    
    pointCam1 = np.array(pointCam1)
    pointCam2 = np.array(pointCam2)
    print(pointCam1.shape)
    points4D = cv2.triangulatePoints(T1, T2, np.transpose(pointCam1), np.transpose(pointCam2)) # Triangulation Equation 계산
    
    depthResultPoints = []
    for i in range(pointCam1.shape[0]):
        x = points4D[:, i]
        x /= x[3]
        ret = x[:3]
        depthResultPoints.append(ret)
    depthResultPoints = np.array(depthResultPoints)
    print(depthResultPoints)
    
    retImg1 = imageA[:,:,:]
    retImg2 = imageB[:,:,:]
    i = 0
    for j, m in enumerate(matches):
        if (m.distance > max(2 * min_dist, 30.)):
            continue
        pt1 = pixel2Cam(kp1[m.queryIdx].pt)
        depth1 = depthResultPoints[i][2]
        cv2.circle(retImg1, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])), 2, getColor(depth1), 2)
        
        pt2 = np.expand_dims(np.array(depthResultPoints[i]), axis=1)
        pt2 = R @ pt2 + t
        depth2 = pt2[2]
        cv2.circle(retImg2, (int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])), 2, getColor(depth2), 2)
        i += 1
    
    cv2.imwrite('./data/triangulation1.png', retImg1)
    cv2.imwrite('./data/triangulation2.png', retImg2)
        