# Trajectory

## Features implemented
- Conducting pose estimation on video footage (using MediaPipe)
- Displaying Frames Per Second (FPS)
- Track the trajectory of a particular joint

## Prerequisites
- A video to be used for demonstration (if you use the CPU, it's better to use shorter videos)
- Put your video path in `cap_video = cv2.VideoCapture('video.mp4')`

## Run
```
$ pyenv virtualenv 3.6 <the name of virtualenv>
```


```
$ pip3 install requirements.txt
```


```
$ python3 demo.py
```

## Demo (Trajectories of Right wrist and Left wrist)
<img width="50%" src="result/output_4.gif"/>