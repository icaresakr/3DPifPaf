###
# Parameters defintions for 3Dpifpaf
#
# NeuroRestore
###

LOAD_DIR = r"/home/vita-w11/Desktop/Ics/3Dpifpaf_r/" #directory of the bag files

SAVE_DIR = r"/home/vita-w11/Desktop/Ics/3Dpifpaf_r/df/" #directory of the saved files

BAG_NAME = r"cam2_record_27_05_2021__13_08_49.bag" #bag file name, don't care about it if running the batch

RUN_BATCH = True # True if convert all bag files inside LOAD_DIR

SAVE_ANNOTATED_VIDEO = False # export the rgb and depth video with overlayed pifpaf keypoints.

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

CAMERA = { # camera recording parameter, attention when running batch if different bags have different resolutions, it cannot be done
    'resolution': [640, 480],
    'fps': 30
}
