import math
#FIXME: input config for camera resolution!
def getValidDepth(center_x,center_y,depth_map,R_max = 15, prev_val = 0):
    
    calc_depth = 0
    for R_s in range(0, R_max):
        for y in range(-R_s,R_s):
            y_r = min(max(center_y + y,0),479)
            for x in range(-R_s,R_s):
                x_r = min(max(center_x + x,0),639)
                calc_depth = depth_map.get_distance(x_r,y_r)

                if calc_depth > 0.5 and calc_depth < 4:
                    return calc_depth


    return prev_val
