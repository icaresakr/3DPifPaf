from tkinter import Y #?
import numpy as np
from math import *
import random
import matplotlib.pyplot as plt
import copy

"""
How confident is PifPaf's conidence score?

How about kalman filter with constant velocity model?

"""

class WeightedAvg:

    def __init__(self, win_size=3, thresh={'nose': 0.07, 'left_eye':0.07, 'right_eye':0.07, 'left_ear':0.07, 'right_ear':0.07,
                        'left_shoulder':0.07, 'right_shoulder':0.07, 'left_elbow':0.07, 'right_elbow':0.07, 
                        'left_wrist':0.07, 'right_wrist':0.07, 'left_hip':0.07, 'right_hip':0.07, 
                        'left_knee':0.07, 'right_knee':0.07, 'left_ankle':0.07, 'right_ankle':0.07}, type = "quadratic", jump_detector = True, jump_penalty = 0.01):

        self.win_size = win_size
        self.thresh = thresh
        self.type = type
        self.jump_detector = jump_detector

        if self.jump_detector:
            self.jump_penalty = jump_penalty

            # PLEASE FIXME
            self.jump_detected = {'nose': False, 'left_eye':False, 'right_eye':False, 'left_ear':False, 'right_ear':False,
                            'left_shoulder':False, 'right_shoulder':False, 'left_elbow':False, 'right_elbow':False, 
                            'left_wrist':False, 'right_wrist':False, 'left_hip':False, 'right_hip':False, 
                            'left_knee':False, 'right_knee':False, 'left_ankle':False, 'right_ankle':False}

            self.jump_count = {'nose': 0, 'left_eye':0, 'right_eye':0, 'left_ear':0, 'right_ear':0,
                            'left_shoulder':0, 'right_shoulder':0, 'left_elbow':0, 'right_elbow':0, 
                            'left_wrist':0, 'right_wrist':0, 'left_hip':0, 'right_hip':0, 
                            'left_knee':0, 'right_knee':0, 'left_ankle':0, 'right_ankle':0}

            self.first = {'nose': True, 'left_eye':True, 'right_eye':True, 'left_ear':True, 'right_ear':True,
                        'left_shoulder':True, 'right_shoulder':True, 'left_elbow':True, 'right_elbow':True, 
                        'left_wrist':True, 'right_wrist':True, 'left_hip':True, 'right_hip':True, 
                        'left_knee':True, 'right_knee':True, 'left_ankle':True, 'right_ankle':True}

            self.segment_center = {'nose': 0, 'left_eye':0, 'right_eye':0, 'left_ear':0, 'right_ear':0,
                            'left_shoulder':0, 'right_shoulder':0, 'left_elbow':0, 'right_elbow':0, 
                            'left_wrist':0, 'right_wrist':0, 'left_hip':0, 'right_hip':0, 
                            'left_knee':0, 'right_knee':0, 'left_ankle':0, 'right_ankle':0}

        #win_size should be an odd number

    def __detect_jump(self, zarray, j_type):

        if self.jump_detector:

            if self.first[j_type]:
                self.segment_center[j_type] = zarray[-1] #last confident joint, oups
                self.first[j_type] = False

            
            jump = self.segment_center[j_type] - zarray[-1]

            if jump > self.thresh[j_type] or jump < -self.thresh[j_type]: #not abs because different thresh
                self.jump_detected[j_type] = True
                #print("detected")
                jump_val = jump
                self.jump_count[j_type] +=1
            
            else:
                self.jump_detected[j_type] = False
                self.jump_count[j_type] = 0
            
            if self.jump_count[j_type] > 100: #FIXME: proportional to keypoint extraction rate
                # check if variance of the signal is > 0 + espi##
                self.jump_detected[j_type] = False
                self.jump_count[j_type] = 0

            if not self.jump_detected[j_type]:
                self.segment_center[j_type] = zarray[-1] #last confident joint, oups

            return(jump, self.jump_detected[j_type]) #FIXME
        
        else:
            return None, False

    def weighted_avg(self, xarray, yarray, zarray, probas, j_type = "right_elbow"): #FIXME: in rep

        window = np.ones(self.win_size)
        filt_x = xarray
        filt_y = yarray
        filt_z = zarray

        epsi=0.1
        probas_temp = probas

        if np.sum(probas_temp)!=0:
            if self.type == "linear":
                window = probas_temp/np.sum(probas_temp)
                print(window)
            
            elif self.type=="quadratic":
                window = np.power(probas_temp,2)/np.sum(np.power(probas_temp,2))
            
            filt_x[-1] = np.dot(filt_x, window) #filter from past
            filt_y[-1] = np.dot(filt_y, window)

            #penalize window from the jumps
            jump_val, detected = self.__detect_jump(filt_z, j_type)
            if detected:
                #print(probas_temp[-1])
                #print(probas_temp)
                probas_temp[-1] *= self.jump_penalty*abs(jump_val)
                #print(probas_temp)
                if self.type == "linear":
                    window = probas_temp/np.sum(probas_temp)
            
                elif self.type=="quadratic":
                    window = np.power(probas_temp,2)/np.sum(np.power(probas_temp,2))
                    #print(window)
            
            filt_z[-1] = np.dot(filt_z, window)

            if probas_temp[-1] < epsi:
                probas_temp[-1] = probas_temp[-2] #propagate the proba?
                
        
        return filt_x[-1], filt_y[-1], filt_z[-1], probas_temp[-1]

pass

if __name__ == "__main__":
    filt = WeightedAvg(win_size=5, jump_detector=True)

    nb_samples = 3000
    temp_len = 20

    noise = [0.02*random.uniform(-1, 1) for i in range(nb_samples)]

    x = [0.5*sin(i) for i in np.linspace(0,temp_len,nb_samples)]
    x = [x[i] + noise[i] for i in range(len(x))]

    probas = [1/(1+2*abs(noise[i])) for i in range(nb_samples)] #proba depends on noise


    jump_mask_l = random.randint(0, nb_samples-1)
    jump_mask_h = random.randint(jump_mask_l, nb_samples - 1)
    x[jump_mask_l:jump_mask_h] = [2 for i in range (jump_mask_h - jump_mask_l +1)]

    probas_corr = copy.copy(probas)
    probas_corr[jump_mask_l:jump_mask_h] = [0. for i in range (jump_mask_h - jump_mask_l +1)]


    history = [[], []]
    history_corr = [[], []]

    y = copy.copy(x)
    _y = copy.copy(x)
    for i in range(nb_samples):
        history[0].append(x[i])
        history[1].append(probas[i])

        history_corr[0].append(x[i])
        history_corr[1].append(probas_corr[i])
        #some processing

        if len(history[0]) == filt.win_size:
            filt.weighted_avg(history_corr[0], history_corr[0], history[0], history[1], "right_elbow")
            #filt.weighted_avg("any", history_corr[0], history_corr[0], history_corr[0], history_corr[1])
            y[i] = history[0][-1]
            #_y[i] = history_corr[0][-1]
            history[0].pop(0)
            history[1].pop(0)
            history_corr[0].pop(0)
            history_corr[1].pop(0)



    #fig = plt.figure()

    plt.plot(x)
    plt.plot(y)
    plt.show()


    """
    xarray = np.array([1.,2.,3.,4.,5.])
    yarray = np.ones(5)
    zarray = np.ones(5)
    probas = np.array([0.,0.,0.5,1.,1.])

    print(xarray)
    print(yarray)
    print(zarray)
    print(probas)

    print(filt.weighted_avg("hand", xarray, yarray, zarray, probas))
    print(xarray, yarray, zarray, probas)
    """