# 3DPifPaf
3D extention of OpenPifPaf pose estimator, using an RGB-d Intel Realsense camera. 
Working on the Linux beast with the GeForce RTX3090.

## 1) On the Linux machine, go to the 3Dpifpaf folder
```bash
cd .../desktop/ICS/3Dpifpaf
```

## 2) Activate the 3Dpifpaf virtual environment
```bash
source 3Dpifpaf/bin/activate
```

## 3) Run 3DPifPaf
### 1) The configuration file [config.py](https://github.com/icaresakr/3dPifPaf/config.py)
Define the parameters of the pose estimation.

```python
LOAD_DIR = r"/home/vita-w11/Desktop/Ics/3Dpifpaf_r/" #directory of the bag files

SAVE_DIR = r"/home/vita-w11/Desktop/Ics/3Dpifpaf_r/df/" #directory of the saved dataframes

BAG_NAME = r"cam2_record_27_05_2021__13_08_49.bag" #bag file name, don't care about it if running the batch

PRE_PROCESS = {
    'selected': ['temporal_filter', 'hole_filling_filter'],
    'hole_filling_filter': {'holes_fill': 2},
    'temporal_filter': {'filter_smooth_alpha': 0.4, 'filter_smooth_delta': 20}
}


POST_PROCESS = {
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

### 2) Run the 3D pose etimation:
#### 2.a) For a single recording 
To run 3DPifPaf for a single intel realsense camera recording (saved as a rosbag (.bag) file), set the ```python RUN_BATCH``` parameter to False in the config.py file, then run the script as follow:
```bash
python /path/to/3dpifpafRep.py /path/to/config.py
```

#### 2.b) For multiple recordings (batch)
To run 3DPifPaf for all recordings located in the LOAD_DIR folder, set the ```python RUN_BATCH``` parameter to True in the config.py file, then run the script as follow:
```bash
python /path/to/3dpifpafRep.py /path/to/config.py
```

