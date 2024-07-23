from flask import Flask, render_template, Response, request, redirect, jsonify
from classifier_model import camera  # Import the camera module
import speech_recognition as sr
import logging

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('home.html')

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to previously detected letter
@app.route('/detected_letter')
def detected_letter():
    letter = next(letter_generator)
    return jsonify({'letter': letter})

@app.route('/transcription', methods=['GET','POST'])
def transcribe():

    transcript_text = ""
    if request.method == 'POST':
        # null file upload case
        if "audio_file" not in request.files:
            return redirect(request.url)
        
        audio_file = request.files["audio_file"]
        # empty file upload case
        if audio_file.filename == "":
            return redirect(request.url)
        
        if audio_file:
            #convert to proper format and make library-wrapped API call
            recognizer = sr.Recognizer()
            sr_audio_file = sr.AudioFile(audio_file)
            with sr_audio_file as source:
                data = recognizer.record(source)
                transcript_text = recognizer.recognize_google(data)

    return render_template('transcription.html', transcript_text=transcript_text)

def gen_video():
    for frame, letter in camera.gen_frames():
        yield frame

# Generator for detected letters
letter_generator = (letter for _, letter in camera.gen_frames())

if __name__ == '__main__':
    app.logger.setLevel(logging.ERROR)
    app.run(debug=True)
