import cv2
import math
import numpy as np

def thresholdIntegral(inputMat,s,T = 0.15):
    
    outputMat=np.zeros(inputMat.shape)
    nRows = inputMat.shape[0]
    nCols = inputMat.shape[1]
    S = int(max(nRows, nCols) / 8)

    s2 = int(S / 4)

    for i in range(nRows):
        y1 = i - s2
        y2 = i + s2

        if (y1 < 0) :
            y1 = 0
        if (y2 >= nRows):
            y2 = nRows - 1

        for j in range(nCols):
            x1 = j - s2
            x2 = j + s2

            if (x1 < 0) :
                x1 = 0
            if (x2 >= nCols):
                x2 = nCols - 1
            count = (x2 - x1)*(y2 - y1)

            sum=s[y2][x2]-s[y2][x1]-s[y1][x2]+s[y1][x1]

            if ((int)(inputMat[i][j] * count) < (int)(sum*(1.0 - T))):
                outputMat[i][j] = 255

    return outputMat


if __name__ == '__main__':
    ratio=1
    image = cv2.imdecode(np.fromfile('image.JPG', dtype=np.uint8), 0)
    
    
    img = cv2.resize(image, (int(image.shape[1] / ratio), int(image.shape[0] / ratio)), cv2.INTER_NEAREST)
    roii = cv2.integral(img)
    
    for j in range(1):
        thresh = thresholdIntegral(img, roii)
    cv2.namedWindow('fast inergral threshold',0)
    cv2.imshow('fast inergral threshold',thresh)
    cv2.imwrite("image.jpg",image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
