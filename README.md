# IphoneSLAM

## How to Run

From the project root, run the following command.
```bash
python3 sfm.py
```

Alterately, use the iPython/jupyter notebook to use live video.
First, open jupyter/Ipython notebooks with 
```bash 
jupyter notebook
```
Then, perform camera calibration by downloading and printing out a checkerboard image (easily obtainable from Google). Then, open the liveCalibrate.ipynb notebook, and run it to begin calibrating the live camera. Rotate the chessboard in front of the camera, trying to orient in various ways (be patient). Eventually, the checkerboard should be seen enough times, and the calibration can proceed. The program will then write out a camera intrinsics matrix to a file for use in the live video SFM.
Next, open up SFM.ipynb in notebook, and prepare to take the video. Once you are ready, select Run all under Cells, and the recording should soon begin. After about 300 frames (usually 10 seconds), the video will end, and the computations of the locations visited will proceed, eventually being displayed in a graph after a short time. If you wish to take another video, you will have to forcefully restart the kernel unfortunately, as opencv3.2 has issues with threading.
