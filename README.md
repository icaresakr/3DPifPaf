# 3DPifPaf
3D extention of OpenPifPaf pose estimator, using an RGB-d Intel Realsense camera. 
For now, you can run it on the Linux beast with the GeForce RTX3090 at ~26 fps when visualization is off, and ~22 fps when on. Otherwise, frame rate will be limited by your computer's processing speed.

## 1) Activate the 3Dpifpaf virtual environment:
#### On a terminal, change the working directory to the 3Dpifpaf folder:
```bash
cd .../desktop/ICS/3Dpifpaf
```

#### Activate the environment with:
```bash
source 3Dpifpaf/bin/activate
```

## 2) Run 3DPifPaf
### i) The configuration file [config.py](https://github.com/icaresakr/3DPifPaf/blob/main/config.py)
Define the parameters of the pose estimation.

```python
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

```

### ii) Run the 3D pose etimation:
#### ii.a) For a single recording 
To run 3DPifPaf for a single recording (saved as a rosbag (.bag) file), set the ```RUN_BATCH``` parameter to ```False``` in the config.py file, and enter the directory of the folder containing the recording in ```LOAD_DIR```, the recording name in ```BAG_NAME```, then run the script as follow:
```bash
python /path/to/3dpifpafRep.py /path/to/config.py
```

#### ii.b) For multiple recordings (batch)
To run 3DPifPaf for all recordings located in the ```LOAD_DIR``` folder, set the ```RUN_BATCH``` parameter to ```True``` in the config.py file, then run the script as follow:
```bash
python /path/to/3dpifpafRep.py /path/to/config.py
```

## 3) The extracted keypoints

Extracted anatomical keypoints are saved as a .csv file in the folder specified by the ```SAVE_DIR``` parameter. The saved dataframes file has the same name as the corresponding recording .bag file. For each body keypoint, the 3D coordinates in meters (relative to camera frame) are extracted as well as the confidence score predicted by OpenPifPaf and the corresponding frame timestamp. 

The camera frame is defined below:

![camera_frame](https://github.com/icaresakr/3DPifPaf/blob/main/images/camera_frame.png?raw=true)



