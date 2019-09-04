import numpy as np
import pandas as pd
import cv2
import time
import os
import glob
import threading
import argparse
from datetime import datetime
from ObjectDetector import *

# colors
green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)
yellow = (255, 255, 0)
purple = (128, 0, 128)
navy = (0, 0, 128)
black = (0, 0, 0)
white = (255, 255, 255)
grey = (128, 128, 128)
silver = (192, 192, 192)


parser = argparse.ArgumentParser(description='Detect a dog, cat, or person')
parser.add_argument('label', choices = ['cat', 'dog', 'person'], default = 'person')
args = parser.parse_args()
label = args['label']


class TimerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self._keepgoing = True
        self.timer_active = False
        
    def run(self):
        while(self._keepgoing):
            if self.timer_active:
                time.sleep(30)
                self.timer_active = False
    
    def stop(self):
        self._keepgoing = False


def detect(region_frame, detector, label):
    _, width, height, channels = detector.engine.get_input_tensor_shape()
    detection_input = cv2.resize(region_frame, (width, height))
    objects = detector.detect(detection_input)

    for obj in objects:
        if obj['label'] == label:
            return True
    
    return False


def select_region(frame):
    (x,y,w,h) = cv2.selectROI("Frame", frame, False, True)
    cv2.destroyAllWindows()
    return (x,y,w,h)

def main():
    detector = ObjectDetector()
    detector.init()
    
    cap = cv2.VideoCapture(0)
    
    timer_thread = TimerThread()
    timer_thread.start()
    
    overlay_color = green
    detection_color = red
    opacity = 0.2
    
    box_color = white
    box_thickness = 1
    
    #buffer to make avoid false reads, make sure we see a person
    #in 10 consecutive frames
    buffer =0
    
    timer_running = False 
    
    ret, initial_frame = cap.read()
    region = select_region(initial_frame)

    current_datetime = datetime.now()
    csv_filename = current_datetime.strftime("%d-%m-%Y")
    #check for security log folder and 
    #folder for today's date otherwise create it
    try:
        os.makedirs('security_logs')
    except OSError:
        pass
        
    try:
        os.makedirs('security_logs/%s' % csv_filename)
    except OSError:
        pass
    
    #check for csv for today
    csv_path = "security_logs/%s/%s.csv" % (csv_filename, csv_filename)
    files = glob.glob(csv_path)
    print("csv_path: %s" % (csv_path))

    #if csv exists, open it into a pandas dataframe. 
    #if no csv, create new dataframe to keep track of events
    #create variable to keep track of event number for the day
    if len(files) == 1:
        print("csv found, loading dataframe")
        df = pd.read_csv(csv_path)
        event = len(df)
    elif len(files) == 0:
        print("no csv found, creating new dataframe")
        df = pd.DataFrame(columns = ['event', 'time'])
        event = 0
    else:
        raise Exception("multiple csv's found matching date")
        
    #it is expensive to constantly append to Pandas dataframe,
    #so while program is running keep events in a list
    event_list = [[]]
    

    while(True):
        ret, input_frame = cap.read()
        x1 = region[0]
        y1 = region[1]
        x2 = x1 + region[2]
        y2 = y1 + region[3]
        
        overlay = input_frame.copy()
        output = input_frame.copy()
        
        region_frame = input_frame[y1:y2, x1:x2]    
        
        found = detect_people(region_frame, detector) 
        
        if found:
            color = detection_color
            buffer += 1
        else:
            color = overlay_color
            buffer = 0
        
        
        box = cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
        output = cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0)
        
        #if buffer is over 10, save an alert, then sleep for 30 seconds
        #in order to prevent saving tons of images from one security alert
        if buffer >= 10 and timer_thread.timer_active == False:
            # we will save an image and add an entry to the csv
            timer_thread.timer_active = True
            event += 1
            image_path = "security_logs/%s" % (csv_filename)
            image_filename = "%d.jpg" % event
            cv2.imwrite(os.path.join(image_path, image_filename), input_frame)
            event_list.append([event, time.strftime("%H-%M")])
        
        
        cv2.imshow("Output", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #first event is empty because we initialized the list, so get rid of first entry in list
            del event_list[0]
            df = df.append(pd.DataFrame(event_list, columns=['event', 'time']), ignore_index=True)
            df.to_csv(csv_path, index=False)
            timer_thread.stop()
            timer_thread.join()
            break


if __name__ == '__main__':
    main()
