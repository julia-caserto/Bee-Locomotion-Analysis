# Bee Locomotion Analysis

The Python script provided in this repository automates the analysis of bee locomotion studies by using the OpenCV library to count the number of crossings in a chamber.

## Requirements

The script can be used with a reasonably recent version of Python (3.7+). The only dependencies are `NumPy`, `SciPy`, and `OpenCV`. These can be installed with

```bash
pip install numpy scipy opencv-contrib-python
```

## Usage

The script can be run as follows.

```bash
python3 tracker.py [-h] [-v VIDEO] [-T TRACKER] [-s START_TIME] [-e END_TIME] [-t TAG] [-V VERBOSE]
```

The available options are the following.

```
  -h, --help            show this help message and exit
  -v VIDEO, --video VIDEO
                        Path to input video file
  -T TRACKER, --tracker TRACKER
                        OpenCV object tracker type
  -s START_TIME, --start_time START_TIME
                        Start time for tracking in HH:MM:SS format
  -e END_TIME, --end_time END_TIME
                        End time for tracking in HH:MM:SS format
  -t TAG, --tag TAG     Tag for output data
  -V VERBOSE, --verbose VERBOSE
                        Verbosity toggle
```

The first time the script is run on a video, the start time, end time, tag (indicating bee ID if there are multiple in the same video) and tracker must be specified. We recommend using the `black_blob` tracker, as it worked much more consistently for us, but other trackers may perform better in different setups. A window will appear showing a frame from the middle of the video. You need to click on the four corners of the chamber of interest, so that the algorith only considers that region. Keep in mind that the algorithm assumes that the long axis of the chamber roughly lines up with the vertical axis of the video. Once the region selection is done, the window will close, and another one will open showing the tracking in progress. It will output a video file showing the tracking, a text file recording the time of each crossing, and a settings file that can be used to rerun the tracking without having to specify again the region of interest, the start and end times, and the tracker. To use it, simply run the script with only the video argument and tag, and it will automatically seach for a corresponding settings file. This is useful when tweaking the parameters of a tracker.

### Example usage

```bash
python tracker.py -v data/trial_1_bees_01-16.mp4 -T black_blob -s 1:57 -e 1:05:50 -t bee_02
```