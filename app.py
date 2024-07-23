from flask import Flask, render_template, Response, request, redirect
from classifier_model import camera  # Import the camera module
import speech_recognition as sr

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('home.html')

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(camera.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

if __name__ == '__main__':
    app.run(debug=True)
