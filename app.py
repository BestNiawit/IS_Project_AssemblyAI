import assemblyai as aai
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import nltk
from transformers import pipeline

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
aai.settings.api_key = "8676fae4821e46e88952da5563a89101"
summarizer = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav', 'mp3', 'flac']


def get_summarizer():
    """Lazy-load summarizer only on first use."""
    global summarizer
    if summarizer is None:
        summarizer = pipeline('summarization')
    return summarizer


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
            # Load the summarizer on demand to speed up app startup.
            current_summarizer = get_summarizer()

            raw_text = getattr(transcript, "text", None)
            if not isinstance(raw_text, str):
                error_message = "There was an error processing the file. Please try another audio file."
                return render_template('index.html', error_message=error_message)

            text = raw_text.strip()
            if not text:
                error_message = "There was an error processing the file. Please try another audio file."
                return render_template('index.html', error_message=error_message)

            try:
                # Generate the summary (keep lengths modest to reduce compute/token usage)
                summary = current_summarizer(text, max_length=200, min_length=30, do_sample=False)
                # Extract the summary text
                summary_text = summary[0]['summary_text']
                # Render the response template with the summary text
                return render_template('uploaded_file.html', filename=filename, transcript=text, summary=summary_text)
            except Exception as e:
                print(e)
                error_message = "Failed to summarize the transcript. Please try again with a different file."
                return render_template('index.html', error_message=error_message)
        else:
            error_message = "There was an error processing the file. Please check the file format and try again."
            return render_template('index.html', error_message=error_message)
    return render_template('index.html')


@app.route('/uploaded_file/<filename>')
def uploaded_file(filename):
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        text = getattr(transcript, "text", None) or ""
        return render_template('uploaded_file.html', filename=filename, transcript=text)
    except Exception as e:
        print(e)
        error_message = "There was an error processing the file. Please check the file format and try again."
        return render_template('index.html', error_message=error_message)


if __name__ == "__main__":
    app.run(debug=True, port=8800, use_reloader=False)
