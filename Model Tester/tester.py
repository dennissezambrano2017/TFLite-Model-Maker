# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support import metadata as _metadata
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import time
import sys
import os

# Initialization
file = str(sys.argv[1]) if len(sys.argv) > 1 else ''
if not os.path.isfile(file):
    print('File not found: '+file)
    sys.exit()

base_options = core.BaseOptions(file_name=file)
detection_options = processor.DetectionOptions(max_results=10)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

#Get possible classes from metadata
displayer = _metadata.MetadataDisplayer.with_model_file(file)
classes = str(displayer.get_associated_file_buffer('labelmap.txt'))[2:-1].split('\\n')
classes.pop()
#print(classes) #This prints out all the possible classes from the model.

#Make colors for each possible class
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def getResults(frame, threshold=0.5):
    st = time.time() #start timer
    tf_image = vision.TensorImage.create_from_array(frame) #used for detections

    # Run inference
    detection_result = detector.detect(tf_image).detections

    # Plot the detection results on the input image
    for obj in detection_result:
        #Get detection information
        index= obj.categories[0].index
        score = obj.categories[0].score
        display_name = obj.categories[0].display_name
        category_name = obj.categories[0].category_name

        color = [int(c) for c in COLORS[index]]

        if(score > threshold):
            # Convert the object bounding box from relative coordinates to absolute
            # coordinates based on the original image resolution
            origin_x = obj.bounding_box.origin_x
            origin_y = obj.bounding_box.origin_y
            width = obj.bounding_box.width
            height = obj.bounding_box.height

            xmin = int(origin_x)
            xmax = int(origin_x+width)
            ymin = int(origin_y)
            ymax = int(origin_y+height)
            
            # Draw the bounding box and label on the image
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            # Make adjustments to make the label visible for all objects
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            label = "{}: {:.0f}%".format(category_name, score * 100)
            cv2.putText(frame, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    et = time.time(); #end timer

    #Put elapsed time on the screen
    elapsed_time = 'delay: '+str((et - st) * 1000)+' miliseconds'
    cv2.putText(frame, elapsed_time, (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)

    #Put quit instructions on the frame
    cv2.putText(frame, 'Press q on your keyboard to quit', (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)
    # Return the final image
    return frame


# Create a VideoCapture object
vid = cv2.VideoCapture(0)

while(True): 
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Detect objects in the frame
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    frame = getResults(frame,threshold)
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 