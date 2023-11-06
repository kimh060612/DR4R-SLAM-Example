import numpy as np
from scipy.linalg import cho_factor, cho_solve
import math
import cv2

HALF_PATCH_SIZE = 4
ITERATION = 10

def getPixelValue(img, x, y):
    x = x if x >= 0 else 0
    y = y if y >= 0 else 0
    x = x if x < img.shape[1] - 1 else img.shape[1] - 2
    y = y if y < img.shape[0] - 1 else img.shape[0] - 2
    x, y = int(x), int(y)
    
    xx = x - math.floor(x)
    yy = y - math.floor(y)
    x_a1 = min(img.shape[1] - 1, int(x) + 1)
    y_a1 = min(img.shape[0] - 1, int(y) + 1)
    return (1 - xx) * (1 - yy) * img[y][x] + xx * (1 - yy) * img[y][x_a1] + (1 - xx) * yy * img[y_a1][x] + xx * yy * img[y_a1][x_a1]

class OpticalFlowTracker:
    def __init__(self, img1, img2, kp1, kp2, inverse=True, has_init=False):
        self.img1 = img1
        self.img2 = img2
        self.kp1 = kp1
        self.kp2 = kp2
        self.inverse = inverse
        self.has_init = has_init
        
    def calculateOpticalFlow(self, num):
        sucess = [ False for _ in range(num) ]
        for i in range(num):
            kp = self.kp1[i]
            dx = 0.0
            dy = 0.0
            if self.has_init:
                dx += self.kp2[i].pt[0] - kp.pt[0]
                dy += self.kp2[i].pt[1] - kp.pt[1]
            
            cost = 0
            lastCost = 0
            result = True
            H = np.zeros((2, 2))
            b = np.zeros((2, 1))
            J = np.zeros((2, 1))
            
            for iter in range(ITERATION):
                if not self.inverse:
                    H = np.zeros((2, 2))
                    b = np.zeros((2, 1))
                else :
                    b = np.zeros((2, 1))
                cost = 0
                
                for x in range(-HALF_PATCH_SIZE, HALF_PATCH_SIZE):
                    for y in range(-HALF_PATCH_SIZE, HALF_PATCH_SIZE):
                        error = getPixelValue(self.img1, kp.pt[0] + x, kp.pt[1] + y) - getPixelValue(self.img2, kp.pt[0] + x + dx, kp.pt[1] + y + dy)
                        if not self.inverse:
                            Jx = 0.5 * (getPixelValue(self.img2, kp.pt[0] + dx + x + 1, kp.pt[1] + dy + y) - getPixelValue(self.img2, kp.pt[0] + dx + x - 1, kp.pt[1] + dy + y))
                            Jy = 0.5 * (getPixelValue(self.img2, kp.pt[0] + dx + x, kp.pt[1] + dy + y + 1) - getPixelValue(self.img2, kp.pt[0] + dx + x, kp.pt[1] + dy + y - 1))
                            JArr = np.array([Jx, Jy])
                            JArr = np.reshape(JArr, (2, 1))
                            J = -1.0 * JArr
                        elif iter == 0 :
                            Jx = 0.5 * (getPixelValue(self.img1, kp.pt[0] + x + 1, kp.pt[1] + y) - getPixelValue(self.img1, kp.pt[0] + x - 1, kp.pt[1] + y))
                            Jy = 0.5 * (getPixelValue(self.img1, kp.pt[0] + x, kp.pt[1] + y + 1) - getPixelValue(self.img1, kp.pt[0] + x, kp.pt[1] + y - 1))
                            JArr = np.array([Jx, Jy])
                            JArr = np.reshape(JArr, (2, 1))
                            J = -1.0 * JArr
                        b += -error * J
                        cost += error * error
                        if not self.inverse or iter == 0:
                            H += J @ J.transpose()
                try :
                    up = cho_solve(cho_factor(H), b)
                except: 
                    result = result and False
                    break
                
                if np.isnan(up).any():
                    print("Gradient is NaN!")
                    result = result and False
                    break
                if (iter > 0 and cost > lastCost) :
                    break
                
                dx += up[0]
                dy += up[1]
                lastCost = cost
                result = result or True
                if np.linalg.norm(up) < 1e-3:
                    break
            
            sucess[i] = result
            if result:
                print(dx, dy)
            self.kp2[i] = cv2.KeyPoint(int(kp.pt[0] + dx), int(kp.pt[1] + dy), size=kp.size)
        return sucess