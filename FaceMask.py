import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.video import VideoStream
import imutils
from playsound import playsound

# Load pre-trained face mask detection model
model = load_model(r"C:\Users\neeraja\Downloads\model.h5")

# Initialize the video stream
vs = VideoStream(src=0).start()

# Load sound file for alert
alert_sound = r"C:\Users\neeraja\Downloads\WhatsApp Audio 2024-03-14 at 7.58.18 AM.mpeg"

# Loop over the frames from the video stream
while True:
    # Read the next frame from the video stream
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Preprocess the frame for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    # (this example assumes you have a pre-trained face detection model)
    # Replace this part with your own face detection mechanism if needed
    # This is just an example of how to use a pre-trained face detector from OpenCV
    faceNet = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI, convert it from BGR to RGB channel ordering,
            # resize it to 224x224, and preprocess it
            faceROI = frame[startY:endY, startX:endX]
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2RGB)
            faceROI = cv2.resize(faceROI, (224, 224))
            faceROI = img_to_array(faceROI)
            faceROI = preprocess_input(faceROI)
            faceROI = np.expand_dims(faceROI, axis=0)

            # Pass the face through the model to determine if the face has a mask or not
            (mask, withoutMask) = model.predict(faceROI)[0]

            # Determine the class label and color for the bounding box
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # If no mask detected, play alert sound
            if label == "No Mask":
                playsound(alert_sound)

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Display the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the q key was pressed, break from the loop
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()