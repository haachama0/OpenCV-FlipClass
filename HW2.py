from pickletools import read_uint1
from random import random
from re import L
from this import d
import cv2
from cv2 import threshold
import numpy as np
from matplotlib import pyplot as plt
from tkinter import filedialog
import random
from scipy import rand

save_img = ''
img_name = 'img'
path = ''

refpt = []
cropping = False
cp = False
points = []

#位置
def find_path(_path=''):
    global path
    path = _path


def Simple_Contour():
    global save_img, img_name, path
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, scr_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    save_img = scr_thresh
    cv2.imshow(img_name, scr_thresh)
    contours, hierarchy = cv2.findContours(scr_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_all = cv2.drawContours(image = img, contours= contours, contourIdx= -1, color=(0, 255, 0), thickness=3)
    cv2.imshow('contours', contour_all)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()

def Find_Contour():
    global save_img, img_name, path
    def contour_threshold_callback(val):
        threshold = val
        canny_output = cv2.Canny(gray, threshold, threshold * 2)
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        cv2.imshow('contours', drawing)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3,3))
    cv2.namedWindow(img_name)
    cv2.imshow(img_name, gray)
    max_thresh = 255
    thresh = 100
    cv2.createTrackbar('Threshold:', img_name, thresh, max_thresh, contour_threshold_callback)
    contour_threshold_callback(thresh)
    cv2.waitKey(0)

def Convex_Hull():
    global save_img, img_name, path
    def convex_hull_callback(val):
        threshold = val
        canny_output = cv2.Canny(gray, threshold, threshold * 2)
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            cv2.drawContours(drawing, contours, i, color)
            cv2.drawContours(drawing, hull_list, i, color)
        cv2.imshow('contours', drawing)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3,3))
    cv2.namedWindow(img_name)
    cv2.imshow(img_name, gray)
    max_thresh = 255
    thresh = 100
    cv2.createTrackbar('Threshold:', img_name, thresh, max_thresh, convex_hull_callback)
    convex_hull_callback(thresh)
    cv2.waitKey(0)

def Bounding_Box():
    global save_img, img_name, path
    def bounding_box_callback(val):
        threshold = val
        canny_output = cv2.Canny(gray, threshold, threshold * 2)
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        centers = [None] * len(contours)
        radius = [None] * len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            cv2.drawContours(drawing, contours_poly, i, color)
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(centers[i][1])), (int(boundRect[i][0] + boundRect[i][2])))
            #cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        cv2.imshow('contours', drawing)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3,3))
    cv2.namedWindow(img_name)
    cv2.imshow(img_name, gray)
    max_thresh = 255
    thresh = 100
    cv2.createTrackbar('Threshold:', img_name, thresh, max_thresh, bounding_box_callback)
    bounding_box_callback(thresh)
    cv2.waitKey(0)

def Basic_Operations():
    global save_img, img_name, path
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    cv2.imshow('threshold', thresh)

    erosion_size = 3
    erosion_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))
    erosion = cv2.erode(thresh,erosion_element)
    cv2.imshow('Erosion', erosion)

    dilation_size = 3
    dilation_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_size + 1, 2 * dilation_size + 1),(dilation_size, dilation_size))
    dilation = cv2.dilate(erosion, dilation_element)
    cv2.imshow('Dilation', dilation)

    opening_size = 3
    opening_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * opening_size + 1, 2 * opening_size + 1),(opening_size, opening_size))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, opening_element)
    cv2.imshow('Opening', opening)

    cv2.waitKey(0)

def Advance_Morphology():
    global save_img, img_name, path
    def morph_shape(val):
        if val == 0:
            return cv2.MORPH_RECT
        elif val == 1:
            return cv2.MORPH_CROSS
        elif val == 2:
            return cv2.MORPH_ELLIPSE

    def erosion_callback(val):
        erosion_size = val
        erosion_shape = morph_shape(cv2.getTrackbarPos(element_shape, erosion_window))
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
        erosion_result = cv2.erode(img, element)
        cv2.imshow(erosion_window, erosion_result)

    def dilation_callback(val):
        dilation_size = val
        dilation_shape = morph_shape(cv2.getTrackbarPos(element_shape, dilation_window))
        element = cv2.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1), (dilation_size, dilation_size))
        dilation_result = cv2.dilate(img, element)
        cv2.imshow(dilation_window, dilation_result)

    max_element_size = 2
    max_kernel_size = 21
    element_shape = 'Element:\n 0: Rectangle\n 1: Cross\n 2: Ellipse'
    kernel_size = 'Kernel size:\n 2n+1'
    erosion_window = 'Erosion'
    dilation_window = "Dilation"

    img = cv2.imread(path)
    cv2.namedWindow(erosion_window)
    cv2.createTrackbar(element_shape, erosion_window, 0, max_element_size, erosion_callback)
    cv2.createTrackbar(kernel_size, erosion_window, 1, max_kernel_size, erosion_callback)

    cv2.namedWindow(dilation_window)
    cv2.createTrackbar(element_shape, dilation_window, 0, max_element_size, dilation_callback)
    cv2.createTrackbar(kernel_size, dilation_window, 1, max_kernel_size, dilation_callback)
    erosion_callback(0)
    cv2.waitKey(0)