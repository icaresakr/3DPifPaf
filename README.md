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

## Run 3DPifPaf
### 1) The configuration file [config.py](https://github.com/icaresakr/3dPifPaf/config.py)
Define the parameters of the pose estimation.

```python 

```

### 2) Run the automated trainer:
#### 2.a) For a single recording 
To run the optimizer sequentially (no parallalization), run the following command in a terminal:
```bash
python /path/to/automated_trainer.py /path/to/optimization_config.py
```

#### 2.b) For multiple recordings (batch)

