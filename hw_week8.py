#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
from functools import partial
import HW
import HW2
import sys

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.geometry('1000x600')
        # Add a menubar
        self.main_menu = tk.Menu(window)
        #Add Exit program
        window.bind('<Escape>', self.del_window)
        # Add file submenu
        self.file_menu = tk.Menu(self.main_menu, tearoff=0)
        self.file_menu.add_command(label='開啟檔案', command=self.open_file)
        self.file_menu.add_command(label='儲存檔案', command=self.save_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label='離開程式', command=self.window.destroy)
        
        # Add operation submenu 
        self.operation_menu = tk.Menu(self.main_menu, tearoff=0)
        self.operation_menu.add_command(label='投影轉換', command=self.roi_file)
        self.operation_menu.add_command(label='二值化調整', command=self.threshold_file)
        self.operation_menu.add_command(label='投射轉換', command=self.perspective_file)
        self.operation_menu.add_command(label='Simple Contour', command=self.simple_contour)
        self.operation_menu.add_command(label='Find Contour', command=self.find_contour)
        self.operation_menu.add_command(label='Convex Hull', command=self.convex_hull)
        self.operation_menu.add_command(label='Bounding Box', command=self.bounding_box)
        self.operation_menu.add_command(label='Basic Operations', command=self.basic_operations)
        self.operation_menu.add_command(label='Advance Morphology', command=self.advance_morphology)

        # Add submenu to mainmenu 
        self.main_menu.add_cascade(label='檔案', menu=self.file_menu)
        self.main_menu.add_cascade(label='功能', menu=self.operation_menu)
        
        # Add space color conversion 色彩空間轉換
        self.color_menu = tk.Menu(self.operation_menu, tearoff=0)
        self.color_menu.add_command(label='RGB圖', command=self.rgb_file)
        self.color_menu.add_command(label='HSV圖', command=self.hsv_file)
        self.color_menu.add_command(label='Gray圖', command=self.gray_file)
        
        #Add 影像資訊呈現
        self.image_menu = tk.Menu(self.operation_menu, tearoff=0)
        self.image_menu.add_command(label='直方圖', command=self.Histogram_file)
        self.image_menu.add_command(label='直方圖均衡化', command=self.Histogram_Equalization_file)
        
        #Add filter
        self.filter_menu = tk.Menu(self.operation_menu, tearoff=0)
        self.filter_menu.add_command(label='高通濾波圖', command=self.high_filter_file)
        self.filter_menu.add_command(label='低通濾波圖', command=self.low_filter_file)
        self.filter_menu.add_command(label='中值濾波圖', command=self.median_filter_file)
        
        #Add geometry
        self.geometry_menu = tk.Menu(self.operation_menu, tearoff=0)
        self.geometry_menu.add_command(label='縮小50%', command=self.resize_file)
        self.geometry_menu.add_command(label='旋轉90度', command=self.rotate_file)
        self.geometry_menu.add_command(label='平移', command=self.translate_file)
        self.geometry_menu.add_command(label='自訂仿射轉換', command=self.affine_file)
        
        # Add submenu to function
        self.operation_menu.add_cascade(label='色彩空間轉換', menu=self.color_menu)
        self.operation_menu.add_cascade(label='影像資訊呈現', menu=self.image_menu)
        self.operation_menu.add_cascade(label='濾波轉換', menu=self.filter_menu)
        self.operation_menu.add_cascade(label='幾何轉換', menu=self.geometry_menu)
        

    
        # display menu
        self.window.config(menu=self.main_menu, cursor='circle')
        # add a video source
        self.video_source = video_source
        # open video source
        '''self.vid = HW.MyVideoCapture(self.video_source)
        # create a canvas to display the video content
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()
        # create a button to capture the frame
        self.btn_snapshot = tk.Button(window, text='snapshot', width='50', command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        self.delay = 15
        self.update()'''
        # loop True
        self.window.mainloop()
      
    def update(self):
        # get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)
    def snapshot(self):
        # get a frame from video source
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite('test.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    def del_window(self, event):
        self.window.destroy()

    def open_file(self):
        file_path = filedialog.askopenfilename(parent=self.window,filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")),title='Select file')
        if not file_path:
            print('file path is empty')
        else:
            print(file_path)
            HW.find_path(file_path)
            HW2.find_path(file_path)
            
    
    def save_file(self):
        HW.Save()
    def rgb_file(self):
        HW.RGB_CvtColor()
    def hsv_file(self):
        HW.HSV_CvtColor()
    def gray_file(self):
        HW.Gray_CvtColor()
    def Histogram_file(self):
        HW.Show_Color_Histogram()
    def Histogram_Equalization_file(self):
        HW.Opencv_Histogram_Equalization()
    def roi_file(self):
        HW.Roi()
    def threshold_file(self):
        HW.Threshold()
    def high_filter_file(self):
        HW.High_Pass_Filter()
    def low_filter_file(self):
        HW.Low_Pass_Filter()
    def median_filter_file(self):
        HW.Median_Filter()
    def resize_file(self):
        HW.Resize()
    def rotate_file(self):
        HW.Rotate()
    def translate_file(self):
        HW.Translate()
    def affine_file(self):
        HW.Affine()
    def perspective_file(self):
        HW.Perspective_Transform()
    def simple_contour(self):
        HW2.Simple_Contour()
    def find_contour(self):
        HW2.Find_Contour()
    def convex_hull(self):
        HW2.Convex_Hull()
    def bounding_box(self):
        HW2.Bounding_Box()
    def basic_operations(self):
        HW2.Basic_Operations()
    def advance_morphology(self):
        HW2.Advance_Morphology()

App(tk.Tk(), 'OpenCv with Tkinter GUI')


# In[ ]:




