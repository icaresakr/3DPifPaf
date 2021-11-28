# 3DPifPaf
3D extention of OpenPifPaf pose estimator, using an RGB-d Intel Realsense camera. 
Working on the Linux beast with the GeForce RTX3090 at ~26 fps when visualization is off, and ~22 fps when on.

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
LOAD_DIR = r"/home/vita-w11/Desktop/Ics/3Dpifpaf_r/" #directory of the bag files

SAVE_DIR = r"/home/vita-w11/Desktop/Ics/3Dpifpaf_r/df/" #directory of the saved dataframes

BAG_NAME = r"cam2_record_27_05_2021__13_08_49.bag" #bag file name, don't care about it if running the batch

PRE_PROCESS = { # Depth map pre-processing filters
    'selected': ['temporal_filter', 'hole_filling_filter'],
    'hole_filling_filter': {'holes_fill': 2},
    'temporal_filter': {'filter_smooth_alpha': 0.4, 'filter_smooth_delta': 20}
}


POST_PROCESS = { # Keypoints post-processing filters, not implemented here yet.
    'selected': None,
    'weighted_average': {"kernel_size": 3}
}

USE_CUDA = True # well use if available

VISUALIZATION = False # show plots and video

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

### 3) The extracted keypoints

Extracted anatomical keypoints are saved as a .csv file in the folder specified by the ```SAVE_DIR``` parameter. For each body keypoint, the 3D coordinates in meters (relative to camera frame) are extracted as well as the confidence score predicted by OpenPifPaf. 

The camera frame is defined below:

![camera_frame](https://github.com/icaresakr/3DPifPaf/blob/main/images/camera_frame.png?raw=true)



