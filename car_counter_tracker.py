import cv2
import torch
import time
import numpy as np
from norfair import Detection, Tracker , draw_points
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Initialize some variables for timing and iteration count
pTime = cTime = iteration = 0
flag = 1
norfair_detections = []
norfair_detections2 = []

# Load the YOLO model from a local directory
model = torch.hub.load(
    r'C:\python\open_cv\object_detection\yolov5\glove_detector\yolov5',
    'custom',
    path=r'C:\Users\Morvarid\Downloads\best (7).pt',
    source='local'
)

# Load the video capture object
cap = cv2.VideoCapture(r"C:\Users\Morvarid\Downloads\pexels-hervé-piglowski-5649316.mp4")

# Initialize the tracker 
comming_tracker = Tracker(distance_function="euclidean", distance_threshold=100)
going_tracker = Tracker(distance_function="euclidean", distance_threshold=100)

# the points of the detection region of the going cars
points21 = [[719, 753], [715, 658], [717, 557], [726, 485], [742, 455], [873, 450], [928, 482], [1030, 541], [1261, 658], [1343, 701], [1341, 753]]
points21 = np.array(points21)
polygon2 = Polygon(points21)

# the points of the detection regoin of the comming cars
points11 = [[715, 751], [708, 678], [705, 592], [707, 541], [713, 484], [719, 454], [531, 451], [396, 509], [152, 645], [0, 755]]
points11 = np.array(points11)
polygon = Polygon(points11)


# Loop over the video frames until the user presses 'q'
while cv2.waitKey(1) != ord("q"):

    # Read the next frame from the video capture object and make a copy of it
    _, image = cap.read()
    image_copy = image.copy()

    # resize and change the color of the images
    image = cv2.resize(image , None , fx=0.7 , fy=0.7)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = cv2.resize(image_copy , None , fx=0.7 , fy=0.7)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    # Compute the frames per second (FPS) and display it on the image
    cTime = time.time()
    if pTime != 0:
        fps = 1 / (cTime - pTime)
        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    pTime = cTime

    # reshaping the point lists to be able to use them 
    points11 = points11.reshape((-1, 1, 2))
    points21 = points21.reshape((-1, 1, 2))

    # draw polygons on the image with the given points
    image_copy = cv2.fillPoly(image_copy , [points11] , (0 , 255 , 0))
    image_copy = cv2.fillPoly(image_copy , [points21] , (0 , 255 , 255))

    if flag:

        # Run object detection on the image using the YOLO model
        outputs = model(image)

        # Extract the predicted bounding boxes, class labels, and confidence scores from the model outputs
        results = outputs.xyxy[0].numpy()
        boxes = results[:, :4]
        # labels = results[:, 5]
        scores = results[:, 4]

        # clear the norfair lists to avoid from overfitting the lists
        norfair_detections.clear()
        norfair_detections2.clear()

        # Loop over the detections and add them to the list of norfair detections
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            x1, y1, x2, y2 = box.astype(np.int32)

            # if the confindence of the detection is above %60 extract its middel point
            if (score >= 0.6) :
                mid_x = int((x2 - x1) / 2)
                mid_y = int((y2 - y1) / 2)
                point = Point([mid_x + x1, mid_y + y1])

            # check if the point is in the comming region polygon
            if polygon.contains(point):
                norfair_detections.append(Detection(points=np.array([mid_x + x1, mid_y + y1])))

            # check if the point is in the going region polygon
            if polygon2.contains(point):
                norfair_detections2.append(Detection(points=np.array([mid_x + x1, mid_y + y1])))


        # Update the tracker with the new detections and get the tracked objects
        tracked_objects = comming_tracker.update(norfair_detections)
        tracked_objects2 = going_tracker.update(norfair_detections2)
        flag = 0

    # controlling the amout of frames to skip for detection
    if not flag:
        iteration += 1
        if (iteration == 4):
            flag = 1
            iteration = 0

    # Draw the tracked objects on the image and display it
    draw_points(image , tracked_objects , text_size=1)
    draw_points(image , tracked_objects2 , text_size=1)

    cv2.putText(image ,"cars in:" + str(comming_tracker.total_object_count) , [500, 417] , cv2.FONT_HERSHEY_COMPLEX , 1 , (0 , 255 , 0) , 2)
    cv2.putText(image ,"cars out:" + str(going_tracker.total_object_count) , [800, 420] , cv2.FONT_HERSHEY_COMPLEX , 1 , (0 , 255 , 255) , 2)

    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image_copy = cv2.cvtColor(image_copy , cv2.COLOR_BGR2RGB)
    final_image = cv2.addWeighted(image , 0.85 , image_copy , 0.15 , 0.0)

    #show the final mixed image
    cv2.imshow("final" , final_image)

