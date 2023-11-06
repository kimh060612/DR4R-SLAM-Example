from .OpticalFlow import OpticalFlowTracker
import cv2

def singleLayerOpticalFlow(img1, img2, kp1, inverse=True, has_init=False):
    kp1List = []
    kp2List = []
    N = len(kp1)
    for i in range(N):
        if i >= len(kp1):
            kp1List.append(cv2.KeyPoint(0, 0, size=3.0))
        else :
            kp1List.append(kp1[i])
        kp2List.append(cv2.KeyPoint(0, 0, size=3.0))
    
    of = OpticalFlowTracker(img1=img1, img2=img2, kp1=kp1List, kp2=kp2List, inverse=inverse, has_init=has_init)
    result = of.calculateOpticalFlow(N)
    return of, result