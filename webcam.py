import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.saved_model.load('ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model')

# Get the function for performing inference
infer = model.signatures['serving_default']

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR image to RGB before processing.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Preprocess the frame for the model
    input_tensor = tf.convert_to_tensor([rgb_frame], dtype=tf.uint8)
    
    # Perform the detection
    detections = infer(input_tensor)
    
    # Draw detections on the frame
    detection_boxes = detections['detection_boxes'].numpy()[0]
    detection_scores = detections['detection_scores'].numpy()[0]
    detection_classes = detections['detection_classes'].numpy()[0]
    height, width, _ = frame.shape

    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:  # Only consider detections with a score above 0.5
            ymin, xmin, ymax, xmax = detection_boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            # Draw rectangle to frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
