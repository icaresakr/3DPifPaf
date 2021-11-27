# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:03:47 2021

@author: Icare SAKR
"""

## FIXME: allign depth image with color image before getting the depth #FIXED
## FIXME: openpifpaf changes the resolution of the image, so the pixels are slightly different ?

## TODO: create a modular pipeline where we can add more cams to refine the measure in an easy way, triangulation, optimization, se  openpose with multicams, calibrate cams.

"""
    dec_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 3);
    spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.50f);
    spat_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
    spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 15);
    temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f);
    temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20.0f);
    rs2::disparity_transform depth_to_disparity(true);
    rs2::disparity_transform disparity_to_depth(true);
    hole_filter.set_option(RS2_OPTION_HOLES_FILL, 1);

    depth = dec_filter.process(depth);

    depth_to_disparity.process(depth);
    disparity_to_depth.process(depth);
    depth = spat_filter.process(depth);
    depth = hole_filter.process(depth);

    frames = frames.apply_filter(dec_filter).apply_filter(spat_filter).apply_filter(temp_filter).apply_filter(hole_filter);
    auto frames_aligned = align_to_depth.process(frames);
"""


import imp
import sys
import glob
import numpy as np
import pyrealsense2 as rs
import cv2
import PIL
#import requests
import torch
import openpifpaf
#import rgbdto3d

import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
import time
import filters.spatial as utils

if os.name == 'nt':
    # had to do this with the windows computer, to solve openmp problem
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class pifpaf3D:
    def __init__(self, cfg):
        self.cfg = cfg
        self.init_cam_params()
        self.setup_pre_filters()
        self.predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16') #openpifpaf predictor


    def setup_pre_filters(self):
        selected = self.cfg.PRE_PROCESS['selected']
        filters_list = []

        if selected == None:
            return filters_list

        for filt_name in selected:
            filter_object = eval('rs.'+filt_name+'()')
            options = self.cfg.PRE_PROCESS[filt_name].keys()
            for option in options:
                eval('filter_object.set_option(rs.option.'+option+', '+str(self.cfg.PRE_PROCESS[filt_name][option])+')')
            filters_list.append(filter_object)

        self.depth_filters = filters_list

    def init_cam_params(self):
        config = rs.config()
        rs.config.enable_device_from_file(config, self.cfg.LOAD_DIR + self.cfg.BAG_NAME)
        self.pipeline = rs.pipeline()

        config.enable_stream(rs.stream.depth, self.cfg.CAMERA['resolution'][0], self.cfg.CAMERA['resolution'][1], rs.format.z16, self.cfg.CAMERA['fps'])
        config.enable_stream(rs.stream.color, self.cfg.CAMERA['resolution'][0], self.cfg.CAMERA['resolution'][1], rs.format.bgr8, self.cfg.CAMERA['fps'])

        cf = self.pipeline.start(config)
        profile = cf.get_stream(rs.stream.depth)
        self.intr = profile.as_video_stream_profile().get_intrinsics() # Intrisic camera parameters

        align_to = rs.stream.color
        self.alignedFs = rs.align(align_to)

    def process_bag(self):
        
        keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                     'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
                     'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        df = dict()
        for joint in keypoints:
            df[joint+".x"] = []
            df[joint+".y"] = []
            df[joint+".z"] = []
            df[joint+".proba"] = []
        df["timestamp"] = []

        t0 = time.time()
        ts = t0
        max_frame_nb = 0

        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()

                aligned_frames = self.alignedFs.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue

                frame_nb = color_frame.get_frame_number()
                if frame_nb < max_frame_nb:
                    break #FIXME

                max_frame_nb = frame_nb

                print("Realtime FPS: {:f}".format(1/(time.time()-ts)))
                ts = time.time()

                # apply depth preprocessing filters
                for filter in self.depth_filters:
                    depth_frame =  filter.process(depth_frame)
                
                depth_frame = rs.depth_frame(depth_frame)


                # Convert color image to numpy array, then to Pillow image
                color_image = np.asanyarray(color_frame.get_data())
                norm_image = color_image/color_image.max()*255
                norm_image = norm_image.astype('uint8')
                
                pil_img = PIL.Image.fromarray(norm_image).convert('RGB')
                predictions, gt_anns, image_meta = self.predictor.pil_image(pil_img)

                if len(predictions) == 0:
                    cloud3d = []

                else:
                    # We have a prediction
                    cloud3d = []
                    #print(depth_frame.get_distance(int(predictions[0].data[1][0]), int(predictions[0].data[1][1])))
                    for i in range(len(predictions[0].data)):
                        if(int(predictions[0].data[i][1]) < 480):
                            if len(df[predictions[0].keypoints[i]+".x"]): 
                                # not first element
                                cloud3d.append(rs.rs2_deproject_pixel_to_point(self.intr,[predictions[0].data[i][0], predictions[0].data[i][1]], utils.getValidDepth(int(predictions[0].data[i][0]), int(predictions[0].data[i][1]),depth_frame,15,0)) + [predictions[0].data[i][2]])
                            else:
                                # first frame
                                cloud3d.append(rs.rs2_deproject_pixel_to_point(self.intr,[predictions[0].data[i][0], predictions[0].data[i][1]], utils.getValidDepth(int(predictions[0].data[i][0]), int(predictions[0].data[i][1]),depth_frame)) + [predictions[0].data[i][2]])
                        else:
                            cloud3d.append([df[predictions[0].keypoints[i]+".x"][-1],-df[predictions[0].keypoints[i]+".z"][-1],df[predictions[0].keypoints[i]+".y"][-1],predictions[0].data[i][2]])
                

                if len(predictions)>0: #FIXME: 
                    for i in range(len(keypoints)):
                        # I think we want 3dpifpaf to give us the data as it is, then with postprocessing we fix the data.
                        df[predictions[0].keypoints[i]+".x"].append(cloud3d[i][0]) # x
                        df[predictions[0].keypoints[i]+".y"].append(cloud3d[i][2]) # depth
                        df[predictions[0].keypoints[i]+".z"].append(-cloud3d[i][1]) # altitude
                        df[predictions[0].keypoints[i]+".proba"].append(cloud3d[i][3]) # proba

                    df["timestamp"].append(ts-t0)
                
                # Visualize realsense color and depth images
                if self.cfg.VISUALIZATION == True:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    images = np.hstack((color_image, depth_colormap))
                    
                    cv2.namedWindow('Preview 3Dpifpaf', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Preview 3Dpifpaf', images)
        
                    k = cv2.waitKey(1)
                    if k == 113: #q pressed
                        break
            
        finally:
            # save data frame
            df_pd = pd.DataFrame.from_dict(df)
            df_pd.to_csv(self.cfg.SAVE_DIR + "df_"+self.cfg.BAG_NAME[:-4]+".csv", index=False)
            # close pipeline
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        cfg_file = input('Config file name? ')
    else:
        cfg_file = sys.argv[1]
    
    cfg = imp.load_source(cfg_file, cfg_file)

    if not cfg.RUN_BATCH: #single bag file
        _3Dpifpaf = pifpaf3D(cfg)
        _3Dpifpaf.process_bag()
    
    else:
        bag_files = glob.glob(cfg.LOAD_DIR + "/**/*.bag", recursive = True)
        for path in bag_files:
            newPath = path.split(os.sep)
            cfg.BAG_NAME = newPath[-1]
            print("--------------")
            print(cfg.BAG_NAME)
            _3Dpifpaf = pifpaf3D(cfg)
            _3Dpifpaf.process_bag()
            print()

        