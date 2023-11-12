import cv2

class WebcamVideoStream:
    def __init__(self, src=0):
        # Initialize the video camera stream and read the first frame
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise IOError("Cannot open webcam")
        self.ret, self.frame = self.stream.read()

    def start(self):
        # Start the video stream
        while True:
            self.ret, self.frame = self.stream.read()
            if not self.ret:
                break
            cv2.imshow('Webcam', self.frame)

            # 'q' key has been pressed, exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def release(self):
        # Release the camera and close all OpenCV windows
        self.stream.release()
        cv2.destroyAllWindows()

# Use the webcam video stream class
if __name__ == '__main__':
    # Initialize webcam video stream object for USB webcam
    webcam_stream = WebcamVideoStream(src=0)  # Assuming '1' is the USB webcam index

    # Start capturing video
    webcam_stream.start()

    # Release resources
    webcam_stream.release()
