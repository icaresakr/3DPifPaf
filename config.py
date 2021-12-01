###
# Parameters defintions for 3Dpifpaf
#
# NeuroRestore
###

LOAD_DIR = r"C:\Users\yes\Desktop\ICS\recordcams\recs\\" #directory of the bag files

SAVE_DIR = r"C:\Users\yes\Desktop\ICS\recordcams\recs\\" #directory of the saved files

BAG_NAME = r"cam0_911222060790_record_24_11_2021_1417_26.bag" #bag file name, don't care about it if running the batch

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

VISUALIZATION = True # show plots and video

EXPORT_VIDEO = True
EXPORT_FPS = 20

RUN_BATCH = True # True if convert all bag files inside LOAD_DIR

CAMERA = { # camera recording parameter, attention when running batch if different bags have different resolutions, it cannot be done
    'resolution': [640, 480],
    'fps': 30
}
