# This is a computer vision pipeline for instance segmentation for the application
# of leaf counting in rossette plants. 
#
# The pipeline utilises the CV2 library and the PlantCV library. 
# 
# Rough as guts! needs a re-write
#
# Author: Ronan Fraser
# Date: June 2019 



import sys
import cv2
import numpy
from plantcv import plantcv as pcv
from matplotlib import pyplot as plt
from scipy.ndimage import label

def segment_on_dt(a, img):

    #logAnd = plantcv.logical_and(a, img)
    #cv2.imshow('logical', logAnd)
    #cv2.waitKey(0)
    gauss = pcv.gaussian_blur(a, (5,5), 0, 0)
    edges = pcv.canny_edge_detect(a, mask=None, sigma=2, low_thresh=15, high_thresh=40, thickness=2, mask_color=None, use_quantiles=False)
    cv2.imshow('canny', edges)
    cv2.waitKey(0)
    canny_dilate = cv2.dilate(edges, None, iterations=1)
    cv2.imshow("canny dilate", canny_dilate)
    cv2.waitKey(0)



    dilate = cv2.dilate(img, None, iterations=1)

    border = dilate - cv2.erode(dilate, None, iterations=2)
    cv2.imshow('border', border)
    cv2.waitKey(0)
    border = cv2.bitwise_or(border, canny_dilate,mask=dilate)
    cv2.imshow("border", border)
    cv2.waitKey(0)


    dt = cv2.distanceTransform(img, 1, 5)
    cv2.imshow('dist trans', dt)
    cv2.waitKey(0)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    cv2.imshow('minmax', dt)
    cv2.waitKey(0)
    _, dt = cv2.threshold(dt, 45, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', dt)
    cv2.waitKey(0)

    seg = cv2.subtract(dt, border)
    cv2.imshow('thresh - border', seg)
    cv2.waitKey(0)

    seg1 = cv2.erode(seg, None, iterations=2)
    cv2.imshow('erode', seg1)
    cv2.waitKey(0)

    fill_image = pcv.closing(seg1)
    cv2.imshow('Closing', fill_image)
    cv2.waitKey(0)


    lbl, ncc = label(fill_image)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now.
    lbl[border == 255] = 255

    print(ncc)

    lbl = lbl.astype(numpy.int32)
    #cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl


img = cv2.imread(sys.argv[1])
img2 = cv2.imread(sys.argv[2])

#img = img[100:400, 50:400]
#img2 = img2[100:400, 50:400]
cv2.imshow("cropped", img)
cv2.waitKey(0)

#blur = cv2.GaussianBlur(img, (15,15), 0)
#cv2.imshow("blur", blur)
#cv2.waitKey(0)

#img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#cv2.imshow("HSV", img_hsv)
#cv2.waitKey(0)



# Pre-processing.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#plt.hist(blur.ravel(), 256, [0,256]); plt.show()
#img_bin,thres = cv2.threshold(img_gray, 125, 255,cv2.THRESH_BINARY)
#img_bin_array = numpy.array(img_bin)

#cv2.imshow("res", thres)
#cv2.waitKey(0)

#img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, numpy.ones((3, 3), dtype=int))

#cv2.imshow('morph', img_bin)
#cv2.waitKey(0)

result = segment_on_dt(img, img2_gray)
cv2.imshow('result', result)
cv2.waitKey(0)
#result2 = cv2.watershed(img, img2)
#cv2.imshow('result2', result2)
#cv2.waitKey(0)

result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)
cv2.imshow('img', img)
cv2.waitKey(0)
