from LKOF.LKFunc import singleLayerOpticalFlow
import numpy as np
import cv2

if __name__ == "__main__":
    img1 = cv2.imread("./data/LK1.png")
    img2 = cv2.imread("./data/LK2.png")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    gftt = cv2.GFTTDetector_create() 
    kp1 = gftt.detect(gray1, None)
    kp2 = gftt.detect(gray2, None)
    pt1 = cv2.goodFeaturesToTrack(gray1, 50, 0.01, 10)
    pt2 = cv2.goodFeaturesToTrack(gray2, 50, 0.01, 10)
    
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, pt1, None)
    of, result = singleLayerOpticalFlow(gray1, gray2, kp1, inverse=False)
    
    imgdstGN2 = cv2.imread("./data/LK2.png")
    imgdstCV2 = cv2.imread("./data/LK2.png")
    for i in range(len(of.kp2)):
        if result[i]:
            xx, yy = int(of.kp2[i].pt[0]), int(of.kp2[i].pt[1])
            x, y = int(of.kp1[i].pt[0]), int(of.kp1[i].pt[1])
            cv2.circle(imgdstGN2, (xx, yy), 2, (0, 255, 0), 2)
            cv2.line(imgdstGN2, (x, y), (xx, yy), (0, 250, 0))
            
    for i in range(len(nextPts)):
        if status[i, 0] :
            xx, yy = tuple(nextPts[i, 0])
            x, y = tuple(pt1[i, 0])
            cv2.circle(imgdstCV2, (int(xx), int(yy)), 2, (0, 250, 0), 2)
            cv2.line(imgdstCV2, (int(x), int(y)), (int(xx), int(yy)), (0, 250, 0))
    
    cv2.imwrite("./data/LKGN.png", imgdstGN2)
    cv2.imwrite("./data/LKGCV.png", imgdstCV2)
    
    
    
