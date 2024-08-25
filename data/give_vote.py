import os
import cv2
import pickle
import csv
from datetime import datetime
from win32com.client import Dispatch
import time
from sklearn.neighbors import KNeighborsClassifier

def speak(message):
    """Function to use text-to-speech to say a message."""
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(message)

# Path to the background image on your desktop
background_path = r"C:\Users\HP\Desktop\background.png"

# Check if the background image file exists
if os.path.exists(background_path):
    imgBackground = cv2.imread(background_path)
    print(f"Background image loaded from {background_path}.")
else:
    print(f"Warning: Background image file '{background_path}' not found.")
    imgBackground = None

# Initialize video capture
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Unable to access video capture device.")
print("Video capture initialized.")

# Initialize face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create 'data' folder if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')
    print("Created 'data' directory.")

# Load saved face data and labels
try:
    with open('data/name.pkl', 'rb') as f:
        LABLES = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    print("Loaded face data and labels successfully.")
except FileNotFoundError as e:
    raise RuntimeError(f"Data file not found: {e}")

# Initialize the K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABLES)
print("KNN classifier trained.")

# Column names for the votes CSV file
COL_NAMES = ['Adhar_ID', 'Vote', 'Date', 'Time']

def check_if_exists(value):
    """Check if the voter has already voted."""
    try:
        with open("Votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
    except FileNotFoundError:
        print("File not found.")
    return False

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Initialize output variable
    output = None

    for (x, y, w, h) in faces:
        # Crop and resize the face image
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        reshaped_img = resized_img.reshape(1, -1)

        # Predict the label for the detected face
        output = knn.predict(reshaped_img)
        print(f"Predicted label: {output[0]}")

        # Get current date and time
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

        if imgBackground is not None:
            # Resize background to match frame dimensions if needed
            img_bg_height, img_bg_width = imgBackground.shape[:2]
            frame_height, frame_width = frame.shape[:2]

            if img_bg_height != frame_height or img_bg_width != frame_width:
                imgBackground = cv2.resize(imgBackground, (frame_width, frame_height))
                print("Resized background image to match frame dimensions.")

            # Overlay frame on resized background
            if (frame.shape[0] <= imgBackground.shape[0] and
                frame.shape[1] <= imgBackground.shape[1]):
                imgBackground[0:frame.shape[0], 0:frame.shape[1]] = frame
                cv2.imshow('frame', imgBackground)
            else:
                print("Background image is too small to display the frame.")
                cv2.imshow('frame', frame)
        else:
            cv2.imshow('frame', frame)

    # Wait for user input
    k = cv2.waitKey(2)

    if output is not None:
        voter_exists = check_if_exists(output[0])
        if voter_exists:
            speak("You have already voted.")
            print("Voter has already voted.")
            break
        
        # Check which key was pressed
        if k == ord('1'):
            vote = "BJP"
            print("Vote recorded for BJP.")
        elif k == ord('2'):
            vote = "CONGRESS"
            print("Vote recorded for CONGRESS.")
        elif k == ord('3'):
            vote = "NOTA"
            print("Vote recorded for NOTA.")
        else:
            continue
        
        # Record the vote
        speak("Your vote has been recorded.")
        time.sleep(5)
        
        with open("Votes.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not os.path.isfile("Votes.csv"):
                writer.writerow(COL_NAMES)
                print("Created new votes CSV file with column headers.")
            attendance = [output[0], vote, date, timestamp]
            writer.writerow(attendance)
            print(f"Recorded vote: {attendance}")

        speak("Thank you for participating in the election.")
        break
    else:
        print("No face detected. Please make sure your face is in the frame.")
       
# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()
print("Video capture released and all windows closed.")
