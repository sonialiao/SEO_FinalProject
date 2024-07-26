import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

# Constants
index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
mean = 0.485 * 255.
std = 0.229 * 255.

# Create runnable session of the exported model
ort_session = ort.InferenceSession("signlanguage.onnx")

# Initialize video capture for webcam
cap = cv2.VideoCapture(0)

# Load in library helpers for identifying hand location
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Continuous capture frame-by-frame
def draw_hand_box(frame, hand_landmarks):
    h, w, c = frame.shape
    for handLMs in hand_landmarks:
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for lm in handLMs.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
        y_min -= 50
        y_max += 50
        x_min -= 50
        x_max += 50
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,255,255), 5)
        return (x_min, x_max, y_min, y_max)

# Function to generate frames for the video feed
def gen_frames():
    prev = ""
    stable_letter = ""
    consecutive_frames = 0
    stability_threshold = 5
    pause_frames = 0
    pause_threshold = 10
    
    letter = None
    
    while True:
        # get frame from webcam
        ret, frame = cap.read()
        if not ret:
            continue

        # convert to RGB and find hand landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        hand_landmarks = hand_results.multi_hand_landmarks
        if hand_landmarks:

            # get bounding box around hand using landmarks
            x_min, x_max, y_min, y_max = draw_hand_box(frame, hand_landmarks)

            # draw landmarks on hand
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # convert to grayscale and crop to hand bounding box
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            hand_frame = frame_gray[y_min:y_max, x_min:x_max]
            if hand_frame.size == 0:
                continue

            # resize, scale values by standard dev. and shape into right format
            x = cv2.resize(hand_frame, (28, 28))
            x = (x - mean) / std
            x = x.reshape(1, 1, 28, 28).astype(np.float32)

            # get output from onnx runtime session
            y = ort_session.run(None, {'input': x})[0]

            # determine final prediction letter
            index = np.argmax(y, axis=1)
            letter = index_to_letter[int(index)]

            # display on frame
            cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (139,0,139), thickness=5)

        # If no letter is detected, increment the pause frames counter
        if not hand_landmarks:
            pause_frames += 1
        else:
            pause_frames = 0

        # Check for stability in letter detection
        if letter == prev and pause_frames < pause_threshold:
            consecutive_frames += 1
        else:
            consecutive_frames = 0
            
        # If the letter has been stable for a certain number of frames, update the stable letter
        if consecutive_frames >= stability_threshold:
            stable_letter = letter

        if pause_frames >= pause_threshold:
            stable_letter = ""
        
        prev = letter
        
        print(stable_letter, letter, prev, consecutive_frames, pause_frames, flush=True)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n', stable_letter)
