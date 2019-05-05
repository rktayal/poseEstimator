import cv2
import argparse

from imutil import FPS
from pose_estimate import PoseEstimator

if __name__ == "__main__":
    # will take mobilenet_thin as base model and
    # image will be resized to 432x368 before processing
    obj = PoseEstimator()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-frames", type=int, default=200,
            help="# of frames to loop over FPS test")
    args = vars(ap.parse_args())
    
    # grab a pointer to video stream and initialize the counter
    print ("[INFO] sampling frames from webcam")
    stream = cv2.VideoCapture(0)
    fps = FPS().start()
    
    # loop over some frames
    while fps._numFrames < args["num_frames"]:
        (grabbed, frame) = stream.read()
    
        # send the frame for inference
        obj.infer(frame)
        obj.showResults()
    
        # update the fps counter
        fps.update()
    
    # stop the timer and display the information
    fps.stop()
    print ("[INFO] elapsed time : {:.2f}".format(fps.elapsed()))
    print ("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # cleanup
    stream.release()
    cv2.destroyAllWindows()
