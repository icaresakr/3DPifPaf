###
# Parameters defintions for 3Dpifpaf
#
# NeuroRestore
###

LOAD_DIR = r"/home/vita-w11/Desktop/Ics/TRIAL_3DPIFPAF/" #directory of the bag files

SAVE_DIR = r"/home/vita-w11/Desktop/Ics/TRIAL_3DPIFPAF/" #directory of the saved files

BAG_NAME = r"cam1_record_27_05_2021__13_13_25.bag" #bag file name, don't care about it if running the batch

PRE_PROCESS = {
    'selected': ['temporal_filter', 'hole_filling_filter'],
    'hole_filling_filter': {'holes_fill': 2},
    'temporal_filter': {'filter_smooth_alpha': 0.4, 'filter_smooth_delta': 20}
}

POST_PROCESS = { #OCCLUSION FILTERS, probably add use_future argument?
    'selected': ['weighted_avg'], #PUT [None] if no post processing is wanted
    'weighted_avg':{"win_size": 3, 
                    "thresholds": {'nose': 0.07, 'left_eye':0.07, 'right_eye':0.07, 'left_ear':0.07, 'right_ear':0.07,
                                'left_shoulder':0.07, 'right_shoulder':0.07, 'left_elbow':0.07, 'right_elbow':0.07, 
                                'left_wrist':0.07, 'right_wrist':0.07, 'left_hip':0.07, 'right_hip':0.07, 
                                'left_knee':0.07, 'right_knee':0.07, 'left_ankle':0.07, 'right_ankle':0.07},
                    "type": "quadratic",
                    "jump_detector" : True,
                    "jump_penalty" : 0.01}
}

USE_CUDA = True # well use if available

VISUALIZATION = False# show plots and video

EXPORT_VIDEO = False
EXPORT_FPS = 5

RUN_BATCH = True # True if convert all bag files inside LOAD_DIR

CAMERA = { # camera recording parameter, attention when running batch if different bags have different resolutions, it cannot be done
    'resolution': [640, 480],
    'fps': 30
}

