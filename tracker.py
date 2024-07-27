#!/usr/bin/env python3

import os
import argparse

import cv2
import numpy as np
from scipy.spatial import ConvexHull

# Setting to control the threshold used to count a crossing
SCHMITT_TRIGGER_THRESHOLD = 8


class CoordPicker:
    """
    This class is used to store coordinates by clicking on a still frame.
    It is used to select the region corresponding to the chamber of a single bee.
    """
    
    def __init__(self) -> None:
        """
        Initializes the coords list.
        """
        self.coords = []
        
    def click_event(self, event: int, x: int, y: int, flags: int, params: None) -> None:
        """
        When a left mouse click is detected it appends the coordinates
        to the list.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coords.append((x,y))


def get_chamber(filename: str) -> np.typing.NDArray:
    """
    Get the coordinates of the four corners of each bee chamber.
    This is done by showing a frame in the middle of the video
    and letting the user click on the corners.
    """
    # Find the middle frame
    cap = cv2.VideoCapture(filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count//2)
    
    # Show the frame and wait until 4 coordinates are picked
    # or the q key is pressed
    cp = CoordPicker()
    _, frame = cap.read()
    if frame is None:
        raise RuntimeError("Error opening video")
    cv2.imshow("Middle frame", frame)
    cv2.setMouseCallback("Middle frame", cp.click_event)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if len(cp.coords) >= 4:
            break
    
    if len(cp.coords) != 4:
        raise RuntimeError("You need to select exactly 4 coordinates")
    
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1) # This is needed so that the windows are properly destroyed
    
    return np.array(cp.coords)


def parse_time(in_str: str) -> int:
    """
    Parses a string of the form "HH:MM:SS" to an integer number of seconds.
    HH and MM are optional.
    """
    data = in_str.split(":")
    secs = 0
    try:
        secs += int(data[-1])
    except Exception as e:
        raise ValueError("Failed to parse string: {e}")
    if len(data) > 1:
        try:
            secs += int(data[-2])*60
        except Exception as e:
            raise ValueError("Failed to parse string: {e}")
    if len(data) > 2:
        try:
            secs += int(data[-3])*3600
        except Exception as e:
            raise ValueError("Failed to parse string: {e}")
    return secs


class BlackBlobFinder:
    """
    This class is used as a tracker. It simply finds a black blob of the right size in the frame.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the tracker.
        """
        self.bee_pos = None
        self.bee_roi = None
        self.frames_since_last_update = 0
        self.verbose = verbose

    def init(self, frame: np.ndarray, bee_roi = list[int]):
        """
        This does nothing. It's just to imitate the behavior of the other trackers.
        """
        pass

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[int, int] | None]:
        """
        Find the position of the bee on the new frame.
        """
        # Settings to identify blobs
        # These need to be adjusted for your particular setup
        MIN_PIXELS = 50
        MAX_PIXELS = 600
        MIN_HEIGHT = 5
        MAX_HEIGHT = 40
        MIN_WIDTH = 5
        MAX_WIDTH = 40
        GRAY_THRESHOLD = 50

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.medianBlur(gray_img, 5)
        _, binary_img = cv2.threshold(gray_img, GRAY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        _, _, boxes, _ = cv2.connectedComponentsWithStats(binary_img)
        filtered_boxes = []
        for x, y, w, h, pixels in boxes:
            if MIN_PIXELS <= pixels <= MAX_PIXELS and MIN_HEIGHT <= h <= MAX_HEIGHT and MIN_WIDTH <= w <= MAX_WIDTH:
                filtered_boxes.append((x, y, w, h, pixels))
        if self.bee_pos is not None:
            scores = [(abs(x+w//2-self.bee_pos[0])+abs(y+h//2-self.bee_pos[1]), abs(pixels-200)) for x,y,w,h,pixels in filtered_boxes]
        else:
            scores = [abs(pixels-200) for _,_,_,_,pixels in filtered_boxes]
        sorted_boxes = [b for _,b in sorted(zip(scores,filtered_boxes))]
        if len(sorted_boxes) == 0:
            if self.verbose:
                print(f"Bee not found! {self.frames_since_last_update} frames since last update")
            self.frames_since_last_update += 1
            # Always says that it found the bee, which makes other parts of the code simpler
            return True, None
        elif len(sorted_boxes) > 1 and self.verbose:
            print("Multiple bee candidates found! Guessing one...")
        self.bee_roi = sorted_boxes[0][:-1]
        self.bee_pos = self.bee_roi[0]+(self.bee_roi[2]//2), self.bee_roi[1]+(self.bee_roi[3]//2)
        self.frames_since_last_update = 0
        return True, self.bee_roi

def main():
    """
    The main function that runs everything.
    """

    # Construct the argument parser and parge the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",      type=str,  help="Path to input video file")
    ap.add_argument("-T", "--tracker",    type=str,  help="OpenCV object tracker type")
    ap.add_argument("-s", "--start_time", type=str,  help="Start time for tracking in HH:MM:SS format")
    ap.add_argument("-e", "--end_time",   type=str,  help="End time for tracking in HH:MM:SS format")
    ap.add_argument("-t", "--tag",        type=str,  help="Tag for output data", default="debug")
    ap.add_argument("-V", "--verbose",    type=bool, help="Verbosity toggle",    default=False)
    args = vars(ap.parse_args())

    tag = args["tag"]
    verbose = args["verbose"]

    settings_filename =  args["video"].split(".")[0]+f"_{tag}_settings.txt"

    if args["video"] is None:
        print("A video file is required.")
        exit(1)

    input_args = False
    if args["tracker"] is not None or args["start_time"] is not None or args["end_time"] is not None:
        input_args = True
        if args["tracker"] is None or args["start_time"] is None or args["end_time"] is None:
            print("If one argument is specified, you must specify all of them.")
            exit(1)

    # If one doesn't input the parameters, then it tries to read them from a previously created settings file
    if not input_args:
        if not os.path.isfile(settings_filename):
            print("Settings file not found. You must specify the parameters manually.")
            exit(1)
        with open(settings_filename, "r") as f:
            settings_dat = eval(f.readlines()[0])
            args["tracker"] = settings_dat[0]
            args["start_time"] = settings_dat[1]
            args["end_time"] = settings_dat[2]
            chamber_vert = np.array(settings_dat[3])

    try:
        OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.legacy.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.legacy.TrackerTLD_create,
            "medianflow": cv2.legacy.TrackerMedianFlow_create,
            "mosse": cv2.legacy.TrackerMOSSE_create,
            "black_blob": BlackBlobFinder,
        }
        tracker = OBJECT_TRACKERS[args["tracker"]]()
    except Exception as e:
        print(f"There was an error: {e}")
        print("If it seems like it should work, them most likely this is because you installed opencv-python instead of opencv-contrib-python")
        exit(1)

    if verbose and isinstance(tracker, BlackBlobFinder):
        tracker.verbose = True
        verbose = False # Delegate print messages to tracker class

    # Start by gettinig the coordinates of the chamber
    if input_args:
        chamber_vert = get_chamber(args["video"])
        if chamber_vert is None:
            exit(1)
        # write the settings to a file so that it's easy to re-run things later
        with open(settings_filename, "w") as f:
            f.write(
                    f"\"{args['tracker']}\","
                    f"\"{args['start_time']}\","
                    f"\"{args['end_time']}\","
                    f"{chamber_vert.tolist()}\n"
                )
            
    start_time = parse_time(args["start_time"])
    end_time = parse_time(args["end_time"])

    min_x = np.min(chamber_vert[:,0])
    max_x = np.max(chamber_vert[:,0])
    min_y = np.min(chamber_vert[:,1])
    max_y = np.max(chamber_vert[:,1])

    video_res = (max_y-min_y, max_x-min_x)

    # Make a mask to remove pixels outside the chamber
    video_mask = np.empty(shape=video_res+(1,), dtype=np.uint8)
    for i in range(video_res[0]):
        print(f"Generating mask {100*i//video_res[0]}%", end="\r")
        for j in range(video_res[1]):
            c = ConvexHull(chamber_vert.tolist() + [[min_x+j,min_y+i]])
            video_mask[i,j] = (0x00 if 4 in c.vertices else 0xFF)
    print("Generating mask 100%")

    # Open video and go to start time
    cap = cv2.VideoCapture(args["video"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps*start_time)
    end_frame = int(fps*end_time)
    max_frames = end_frame - start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Set up output files
    out_filename =  args["video"].split(".")[0]+f"_{tag}"
    out_data_file = open(out_filename + ".txt", "w")
    out_vid_file = cv2.VideoWriter(out_filename + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (video_res[0],video_res[1]))

    # Set the difference between thresholds
    diff = video_res[0]//SCHMITT_TRIGGER_THRESHOLD

    # Create additional black blob tracker when using opencv trackers so we can find the bee initially
    bee_finder = (tracker if isinstance(tracker, BlackBlobFinder) else BlackBlobFinder())

    bee_in_upper_half = None
    bee_found = False
    bee_pos = None
    bee_roi = None
    n_crossings = 0
    frame_count = 0
    while frame_count < max_frames:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            print("Couldn't read frame")
            break
        # Apply mask to frame
        frame = frame[min_y:max_y,min_x:max_x]
        frame &= video_mask
        frame ^= ~video_mask
        # If bee location is unknown then find it
        if bee_found:
            bee_found, bee_roi = tracker.update(frame)
            if bee_found and bee_roi is not None:
                bee_roi = [int(c) for c in bee_roi]
                bee_pos = bee_roi[0]+(bee_roi[2]//2), bee_roi[1]+(bee_roi[3]//2)
            elif verbose:
                print("Tracker lost bee")
        else:
            bee_finder.bee_roi = bee_roi
            ret, bee_roi = bee_finder.update(frame)
            # If this fails then we don't update the position of the bee
            if bee_roi is not None:
                bee_pos = bee_roi[0]+(bee_roi[2]//2), bee_roi[1]+(bee_roi[3]//2)
            if tracker is not None:
                tracker.init(frame, bee_roi)
                bee_found = True
        # Check if it crossed a threshold
        crossed = False
        if bee_pos is not None:
            if bee_pos[1] > video_res[0]//2 + diff:
                if bee_in_upper_half is not None and bee_in_upper_half:
                    crossed = True
                bee_in_upper_half = False
            elif bee_pos[1] < video_res[0]//2 - diff:
                if bee_in_upper_half is not None and not bee_in_upper_half:
                    crossed = True
                bee_in_upper_half = True
        if crossed:
            n_crossings += 1
            out_data_file.write(f"{frame_count/fps:.2f}\n")
        # Draw a rectangle around the bee
        if bee_roi is not None:
            (x, y, w, h) = [int(v) for v in bee_roi]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Transpose image
        frame = np.swapaxes(frame, 0, 1).copy() # No idea why you need to .copy()
        # Draw the number of crossings
        text = f"Crossings: {n_crossings}"
        cv2.putText(frame, text, (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        # show the output frame and check for keyboard input
        out_vid_file.write(frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    cap.release()
    out_data_file.close()
    out_vid_file.release()

# Run main if the script is executed directly
if __name__ == "__main__":
   main()