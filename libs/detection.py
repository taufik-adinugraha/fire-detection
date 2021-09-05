import numpy as np
import cv2


# initialize minimum probability to filter weak detections along with the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3


def detect_object(frame, net, ln, Idxs):
	(H, W) = frame.shape[:2]
	results = []

	# construct a blob from the input frame and then perform a forward pass of the YOLO object detector
	try:
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		net.setInput(blob)
		layerOutputs = net.forward(ln)
	except:
		return []

	# initialize our lists of detected bounding boxes, centroids, and confidences, respectively
	boxes = []
	classIDs = []
	centroids = []
	confidences = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter detections
			if classID in Idxs and confidence > MIN_CONF:
				# scale the bounding box coordinates back relative to the size of the image
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, centroids, and confidences
				boxes.append([x, y, int(width), int(height)])
				classIDs.append(classID)
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# update our results list
			r = (classIDs[i], confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# return the list of results
	return results



def read_plate(plate_image, net, ln, classes):
	results = detect_object(plate_image, net, ln, range(36))
	all_char = []
	for result in results:
	    classID, prob, bbox, centroid = result
	    (startX, startY, endX, endY) = bbox
	    cY = startY + (endY-startY)//2
	    dY = endY - startY
	    all_char.append((classID, cY, dY, centroid, bbox))
	
	# median of the centers and the widths
	med_cY = np.median([i[1] for i in all_char])
	median_dY = np.mean([i[2] for i in all_char])	
	
	#seperate between plate number (plate_num) and the expiry date (date_num)
	up = med_cY - median_dY
	lw = med_cY + median_dY

	# plate_num: [classID, centroid[0], centroid[1], bbox]
	plate_num = [(i[0], i[3][0], i[3][1], i[4]) for i in all_char if ((i[1] > up) and (i[1] < lw) and (i[2] > 0.75*median_dY))]
	date_num = [(i[0], i[3][0], i[3][1], i[4]) for i in all_char if i[1] > lw]
	
	# sort from left to right
	date_num = sorted(date_num, key=lambda tup: tup[1])
	plate_num = sorted(plate_num, key=lambda tup: tup[1])

	# convert to string
	text = "".join([classes[i[0]] for i in plate_num])
	text1 = "".join([classes[i[0]] for i in date_num])
	
	# separate the plate number: STRINGS NUMBERS STRINGS
	front, number, back = [], [], []
	for i, n in enumerate(list(text)):
	    try:
	        int(n)
	        number.append(n)
	    except ValueError:
	        if len(number) == 0:
	            front.append(n)
	        else:
	            back.append(n)
	front += [' ']
	number += [' ']
	text = front + number + back

	# plate number
	text = "".join([i for i in text])
	# expiry date
	text1 = "".join([i for i in text1])

	# get all characters with the respective bounding boxes
	# chars: [classID, bbox[0], bbox[1], bbox[2], bbox[3], centroid[0], centroid[1]]
	all_char = plate_num + date_num
	chars = [[i[0], i[3][0], i[3][1], i[3][2], i[3][3], i[1], i[2]] for i in all_char]
	
	return text, text1, chars


