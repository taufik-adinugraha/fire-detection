from libs import detection
import numpy as np
import argparse
import imutils
import cv2
import os



USE_GPU = False

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = 'models/classes.txt'
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = 'models/yolo_best.weights'
configPath = 'models/yolov4.cfg'

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = int(vs.get(cv2.CAP_PROP_FPS))


# loop over the frames from the video stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# resize the frame and then detect people (and only people) in it
	# frame = imutils.resize(frame, width=600)
	(H, W) = frame.shape[:2]
		
	objects = ['smoke', 'fire']
	results = detection.detect_object(frame, net, ln, Idxs=[LABELS.index(i) for i in objects if LABELS.index(i) is not None])

	# loop over the results
	for (i, (classID, prob, bbox, centroid)) in enumerate(results):
		print(LABELS[classID])
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid

		if LABELS[classID] == 'smoke':
			text = 'ASAP'
			color = [255, 255, 255]
		else:
			text = 'API'
			color = [0, 255, 255]

		# get the width and height of the text box
		font_scale = 0.7
		font = cv2.FONT_ITALIC
		(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
		# set the text start position
		text_offset_x, text_offset_y = startX + 3, startY + 20
		# make the coords of the box with a small padding of two pixels
		box_coords = ((text_offset_x - 5, text_offset_y + 5), (text_offset_x + text_width + 5, text_offset_y - text_height - 5))
		overlay = frame.copy()
		cv2.rectangle(overlay, box_coords[0], box_coords[1], color, -1)
		cv2.rectangle(overlay, (startX, startY), (endX, endY), color, 3)
		cv2.putText(overlay, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0,0,0), thickness=2)
		# apply the overlay
		alpha=0.6
		cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


	# check to see if the output frame should be displayed to our screen
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output video file
	if writer is not None:
		writer.write(frame)

