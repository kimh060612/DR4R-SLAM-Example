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

def pixel2Cam(p: tuple):
    return (
        (p[0] - K[0][2]) / K[0][0],
        (p[1] - K[1][2]) / K[1][1]
    )


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

    points3D = []
    points2D = []

    for i, con in enumerate(desc1):
        dist = matches[i].distance
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)

    for m in matches:
        if (m.distance > max(2 * min_dist, 30.)):
            continue
        x, y = int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])
        d = imageADepth[y][x]
        if d == 0:
            continue
        dd = d / 5000.0
        p1 = pixel2Cam(kp1[m.queryIdx].pt)
        points3D.append((p1[0] * dd, p1[1] * dd, dd))
        points2D.append(kp2[m.trainIdx].pt)
    
    print(len(points3D))
    success, vector_rotation, vector_translation = cv2.solvePnP(
        np.array(points3D), np.array(points2D), K, distortions, flags=0
    )
    rotationMatrix, _ = cv2.Rodrigues(vector_rotation)
    print(rotationMatrix, vector_rotation, vector_translation, sep="\n")