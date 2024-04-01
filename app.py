import assemblyai as aai
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import nltk
from transformers import pipeline

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
aai.settings.api_key = "8676fae4821e46e88952da5563a89101"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav', 'mp3', 'flac']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path)
        if transcript is not None:
            # Initialize the summarizer
            summarizer = pipeline('summarization')
            # Generate the summary
            summary = summarizer(transcript.text, max_length=250, min_length=30, do_sample=False)
            # Extract the summary text
            summary_text = summary[0]['summary_text']
            # Render the response template with the summary text
            return render_template('uploaded_file.html', filename=filename, transcript=transcript.text,summary=summary_text)
        else:
            error_message = "There was an error processing the file. Please check the file format and try again."
            return render_template('index.html', error_message=error_message)
    return render_template('index.html')


@app.route('/uploaded_file/<filename>')
def uploaded_file(filename):
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('uploaded_file.html', filename=filename, transcript=transcript.text)
    except Exception as e:
        print(e)
        error_message = "There was an error processing the file. Please check the file format and try again."
        return render_template('index.html', error_message=error_message)


if __name__ == "__main__":
    app.run(debug=True, port=8800, use_reloader=False)
