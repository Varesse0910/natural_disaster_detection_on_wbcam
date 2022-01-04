# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
# from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
# from camera import Camera
import argparse
import datetime
import imutils
import time
import cv2
from tensorflow.keras.models import load_model
# from tensorflow.keras.models import load_model
from natural import pred
from natural.pyimagesearch import config
from collections import deque
import numpy as np
import argparse
import cv2
# import h5py

model = load_model(config.MODEL_PATH)
# Q = deque(maxlen=args["size"])
vs = cv2.VideoCapture(0)
# writer = None
(W, H) = (None, None)

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()
label = None
# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_motion(frameCount):
	global vs, outputFrame, lock
	while True:
		frame = vs.read()
		output = frame.copy()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (224, 224))
		frame = frame.astype("float32")

		from natural.pred import predict
		label = predict(frame)
		text = "activity: {}".format(label)
		cv2.putText(frame, text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
					0.35, (0, 255, 0), 1)

		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, 20),
			# "%I:%M:%S%p"), (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 255), 1)
		high = imutils.resize(frame,width=600,height=800)
							  # ,height=600)
		with lock:
			outputFrame = high.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock, label

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, #required=True,
		help="ip address of the device",
					default="0.0.0.0")
	ap.add_argument("-o", "--port", type=int,# required=True,
		help="ephemeral port number of the server (1024 to 65535)",
					default="8000")
	ap.add_argument("-f", "--frame-count", type=int, default=4,#32
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()