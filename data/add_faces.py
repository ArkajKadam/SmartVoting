import cv2
import pickle
import numpy as np
import os

def main():
    # Create data folder if it doesn't exist
    if not os.path.exists('data/'):
        os.makedirs('data/')

    # Initialize video capture and face detection
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not video.isOpened():
        print("Error: Unable to access the video capture device.")
        return

    # Initialize variables for collecting face data
    faces_data = []
    labels = []
    name = input("Enter your Aadhar number: ")
    framesTotal = 51  # Total number of frames to collect
    captureAfterFrame = 2  # Capture frame every N frames
    i = 0  # Frame counter

    print("Collecting face data...")

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Crop and resize the face region
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50))

            # Store face data and label
            if len(faces_data) < framesTotal and i % captureAfterFrame == 0:
                faces_data.append(resized_img)
                labels.append(name)
                print(f"Collected {len(faces_data)} face images.")

            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the collected data count on the screen
        cv2.putText(frame, f"Collected: {len(faces_data)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 2)
        cv2.imshow('frame', frame)

        # Check for user input to quit or if enough frames have been collected
        k = cv2.waitKey(4)
        if k == ord('q'):
            print("Quitting...")
            break
        if len(faces_data) >= framesTotal:
            print("Collected sufficient data.")
            break

    # Release video capture and close all windows
    video.release()
    cv2.destroyAllWindows()

    # Convert lists to numpy arrays
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape((framesTotal, -1))

    # Save collected data to files
    try:
        with open('data/name.pkl', 'wb') as f:
            pickle.dump(labels, f)
        print("Labels saved successfully.")
    except Exception as e:
        print(f"Error saving labels: {e}")

    try:
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
        print("Face data saved successfully.")
    except Exception as e:
        print(f"Error saving face data: {e}")

if __name__ == "__main__":
    main()
