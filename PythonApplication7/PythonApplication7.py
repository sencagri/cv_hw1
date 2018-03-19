import cv2
from array import array
import numpy as np

def mergeImages(img1, img2, scale=1):
    # img's shapes to control image channels, if we have a single channel image we have to extend it to 3 channel img
    img1Shape = len(img1.shape)
    img2Shape = len(img2.shape)

    if(img1Shape==2):
        img1 = cv2. cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if(img2Shape==2):
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # scale the image if scale is set between 0..1
    if(scale > 0 and scale<1):
        img1 = cv2.resize(imga, scale)
        img2 = cv2.resize(imgb, scale)

    result = np.concatenate((img1, img2), axis=1)
    return result

def conv2GrayScaleOpencv(srcImg):
    result = np.copy(srcImg)
    result = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    return result

def conv2GrayScale(srcImg):
    # as suggested in ref link ( ref : http://www.robindavid.fr/opencv-tutorial/chapter3-pixel-access-and-matrix-iteration.html )
    width = srcImg.shape[0]
    height = srcImg.shape[1]
    shape = (width, height)
    
    result = np.zeros(shape, dtype="uint8")

    for i in np.arange(0, width):
        for j in np.arange(0, height):
            val = srcImg[i,j][0] * 0.11 + srcImg[i,j][1] * 0.59 + srcImg[i,j][2] * 0.3
            result[i,j] = val
    return result

def gammaCorr(srcImg, gamma):
    # defining a lookup table for fast swap operation ( ref : https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/ )
    invGamma = 1.0 / gamma
    lut = np.array([((i/255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(srcImg, lut)

def histEqualizeOpencv(srcImg):
    return cv2.equalizeHist(cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY))

def histEqualize(srcImg):
    grayImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist(grayImg, [0], None, [256], [0,256])
    histCDF = np.zeros((1,256), "float")
    for i in np.arange(hist.shape[0]):
        val = 0.0
        for j in np.arange(hist.shape[0]):
            val += hist[j]
        histCDF[i] = val / 256.0
    histLUT = np.array([math.floor(i * 256)])
    return hist

# Read the test image
org = cv2.imread("testimg.png")
org2 = cv2.imread("testimg2.jpg")
org3 = cv2.imread("testimg3.jpg")

org2 = cv2.resize(org2, (0,0), fx=0.25, fy=0.25)
org3 = cv2.resize(org3, (0,0), fx=0.25, fy=0.25)


# Q1 - convert to grayscale of the image by opencv and manually written code
grayImg = conv2GrayScale(org)
grayImgOpencv = conv2GrayScaleOpencv(org)
plot = mergeImages(org,grayImg)
plot2 = mergeImages(org, grayImgOpencv)

#cv2.imshow("Q1 - Original vs. manually written",plot);
#cv2.imshow("Q1 - Original vs. opencv available method",plot2);

# Q2 - apply power law transformation to dark and light images
lightImagewithHighGamma = gammaCorr(org3, 2)
cv2.putText(lightImagewithHighGamma, "gamma:2",(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))

lightImagewithLowGamma = gammaCorr(org3, 0.5)
cv2.putText(lightImagewithLowGamma, "gamma:0.5",(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))

darkImagewithHighGamma = gammaCorr(org2, 2)
cv2.putText(darkImagewithHighGamma, "gamma:2",(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))

darkImagewithLowGamma = gammaCorr(org2, 0.5)
cv2.putText(darkImagewithLowGamma, "gamma:0.5",(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))

plot3 = mergeImages(org3, lightImagewithHighGamma)
plot4 = mergeImages(org3, lightImagewithLowGamma)

plot5 = mergeImages(org2, darkImagewithHighGamma)
plot6 = mergeImages(org2, darkImagewithLowGamma)

equalizedDark = histEqualizeOpencv(darkImagewithHighGamma)
equalizedLow = histEqualizeOpencv(darkImagewithLowGamma)

equalizedDarkManual = histEqualize(darkImagewithHighGamma)
equalizedLowManual = histEqualize(darkImagewithLowGamma)

plot7 = mergeImages(equalizedDark,equalizedLow)
plot7 = mergeImages(org2,plot7)

cv2.imshow("hist equalized image", plot7)

#cv2.imshow("Light image with high gamma",plot3)
#cv2.imshow("Light image with low gamma",plot4)
#cv2.imshow("Dark image with high gamma",plot5)
#cv2.imshow("Dark image with low gamma",plot6)

cv2.waitKey();
cv2.destroyAllWindows();


