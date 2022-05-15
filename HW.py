#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import filedialog

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
    Img()
    
#儲存
def Save():
    global save_img
    cv2.imwrite('output.jpg', save_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
#原圖
def Img():
    global save_img, img_name, path
    cv2.namedWindow(img_name)
    img = cv2.imread(path)
    save_img = img
    cv2.imshow(img_name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()
    
def click_and_crop(event, x, y, flags, param): 
    global refpt, cropping, path, img_name, cp
    srcImg = cv2.imread(path)
    if event == cv2.EVENT_LBUTTONDOWN:
        refpt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        refpt.append((x, y))
        cropping = False
        cv2.rectangle(srcImg, refpt[0], refpt[1], (0, 255, 0), 2)
        cp = True
        
#投影轉換
def Roi():
    global save_img, img_name, path, refpt, cropping, cp
    cp = False
    srcImg = cv2.imread(path) 
    clone = srcImg.copy()
    cv2.namedWindow(img_name,cv2.WINDOW_NORMAL)                 
    cv2.moveWindow(img_name,500,100)
    cv2.setMouseCallback(img_name, click_and_crop)
    while True:
        if cp == True and refpt != None:
            cv2.rectangle(srcImg, refpt[0], refpt[1], (0, 255, 0), 2)
            cp = False
        cv2.imshow(img_name, srcImg)
        #if (cv2.waitKey(1) & 0xFF) == ord('q'):
        #    cv2.destroyWindow(img_name)
        #    break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            srcImg = clone.copy()
        if key == ord("c"):
            break
    if len(refpt) == 2:
        roi = clone[refpt[0][1]:refpt[1][1], refpt[0][0]:refpt[1][0]]
        cv2.imshow('ROI',roi)
        cv2.waitKey(0)
    save_img = roi
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()
        
#直方圖
def Show_Color_Histogram():
    global save_img, img_name, path
    img = cv2.imread(path)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()
    
#空間轉換灰色
def Gray_CvtColor():
    global save_img, img_name, path
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    save_img = gray
    cv2.imshow(img_name,gray)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()
    
#空間轉換HSV  
def HSV_CvtColor():
    global save_img, img_name, path
    img = cv2.imread(path)
    HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    save_img = HSV
    cv2.imshow(img_name,HSV)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()
    
#空間轉換RGB
def RGB_CvtColor():
    global save_img, img_name, path
    img = cv2.imread(path)
    RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    save_img = RGB
    cv2.imshow(img_name,RGB)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()
    

def empty(v):
    pass    

#二值化調整
def Threshold():
    global save_img, img_name, path
    cv2.namedWindow(img_name)
    img_ori = cv2.imread(path)
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    cv2.createTrackbar('Max', img_name, 0, 255, empty)
    cv2.createTrackbar('Ret', img_name, 0, 127, empty)
    while True:
        #_max 最大灰度值
        _max =  cv2.getTrackbarPos('Max', img_name)
        #閥值
        _ret =  cv2.getTrackbarPos('Ret', img_name)
        ret, mask = cv2.threshold(img, _ret, _max, cv2.THRESH_BINARY)
        save_img = mask
        cv2.imshow(img_name,mask)
        if (cv2.waitKey(1) & 0xFF) == 27:
            cv2.destroyWindow(img_name)
            break
    cv2.destroyAllWindows()
    
#直方圖均衡化
def Opencv_Histogram_Equalization():
    global save_img, img_name, path
    cv2.namedWindow(img_name)
    img = cv2.imread(path)
    hist_ori = cv2.calcHist([img], [0], None, [256], [0,256])
    plt.figure(1)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    R, G, B = cv2.split(img)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    equ = cv2.merge((output1_R, output1_G, output1_B))
    save_img = equ
    cv2.imshow(img_name,equ)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()
    
#低通濾波
def Low_Pass_Filter():
    global save_img, img_name, path
    cv2.namedWindow(img_name)
    img_ori = cv2.imread(path)
    kernel = np.array([
                [1.,1.,1.],
                [1.,1.,1.],
                [1.,1.,1.]
                ])
    img = cv2.filter2D(img_ori, -1, kernel)
    save_img = img
    cv2.imshow(img_name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()
    
#高通濾波
def High_Pass_Filter():
    global save_img, img_name, path
    cv2.namedWindow(img_name)
    img_ori = cv2.imread(path)
    kernel = np.array([
            [-1,-1,-1],
            [-1, 8,-1],
            [-1,-1,-1],
            ])
    img = cv2.filter2D(img_ori, -1, kernel)
    save_img = img
    cv2.imshow(img_name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()
    
#中值濾波
def Median_Filter():
    global save_img, img_name, path
    cv2.namedWindow(img_name)
    img_ori = cv2.imread(path)
    img = cv2.medianBlur(img_ori, 11)
    save_img = img
    cv2.imshow(img_name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()

#修改大小
def Resize():
    global save_img, img_name, path
    cv2.namedWindow(img_name)
    img_ori = cv2.imread(path)
    img = cv2.resize(img_ori,(0,0), fx = 0.5, fy = 0.5)
    save_img = img
    cv2.imshow(img_name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()

#旋轉
def Rotate():
    global save_img, img_name, path
    cv2.namedWindow(img_name)
    img_ori = cv2.imread(path)
    rows, cols, ch = img_ori.shape
    #轉90度
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
    img = cv2.warpAffine(img_ori, M, (cols, rows))
    save_img = img
    cv2.imshow(img_name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()

#平移
def Translate():
    global save_img, img_name, path
    cv2.namedWindow(img_name)
    img_ori = cv2.imread(path)
    rows, cols, ch = img_ori.shape
    M = np.float32([[1, 0, 100],
                    [0, 1, 50]])
    img = cv2.warpAffine(img_ori, M, (cols, rows))
    save_img = img
    cv2.imshow(img_name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(img_name)
    cv2.destroyAllWindows()

#仿射轉換
def Affine():
    global save_img, img_name, path
    cv2.namedWindow(img_name)
    img_ori = cv2.imread(path)
    rows, cols, ch = img_ori.shape
    cv2.createTrackbar('A00', img_name, 200, 200, empty)
    cv2.createTrackbar('A01', img_name, 100, 200, empty)
    cv2.createTrackbar('B02', img_name, 0, cols, empty)
    cv2.createTrackbar('A10', img_name, 100, 200, empty)
    cv2.createTrackbar('A11', img_name, 200, 200, empty)
    cv2.createTrackbar('B12', img_name, 0, rows, empty)
    while True:
        A00 =  cv2.getTrackbarPos('A00', img_name)
        A01 =  cv2.getTrackbarPos('A01', img_name)
        B02 =  cv2.getTrackbarPos('B02', img_name)
        A10 =  cv2.getTrackbarPos('A10', img_name)
        A11 =  cv2.getTrackbarPos('A11', img_name)
        B12 =  cv2.getTrackbarPos('B12', img_name)
        M = np.float32([[A00/100-1, A01/100-1, B02],
                        [A10/100-1, A11/100-1, B12]])
        img = cv2.warpAffine(img_ori, M, (cols, rows))
        save_img = img
        cv2.imshow(img_name,img)        
        if (cv2.waitKey(1) & 0xFF) == 27:
            cv2.destroyWindow(img_name)
            break
    cv2.destroyAllWindows()
    
def OnMouseAction(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_back,(x, y), 3, (0,0,255), -1)
        cv2.imshow(img_name,img_back)
        points.append([x, y])
        if len(points) == 4:
            pts1 = np.float32(points)
            pts2 = np.float32(
                    [[0,0], [400, 0], [400,400], [0, 400]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img_back, M, (400,400))
            save_img = img
            cv2.imshow(img_name,img)
            points.clear()
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
def Perspective_Transform():
    global save_img, img_name, path
    global img_back, img_clone
    img_ori = cv2.imread(path)
    img_back = img_ori
    img_clone = img_ori
    cv2.namedWindow(img_name,cv2.WINDOW_NORMAL)   
    cv2.imshow(img_name,img_ori)
    cv2.setMouseCallback(img_name, OnMouseAction) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#攝影
class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError('Unable to open video source', video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None


# In[ ]:




