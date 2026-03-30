# AssemblyAI-Transcriber
## Speech-to-Text Transcription and Summarization
This project demonstrates a Flask web application that allows users to upload audio files and receive a transcript and summary of the audio content.

# Requirements
Python 3.8 or higher
Flask 2.0 or higher
assemblyai 0.2.0 or higher
transformers 4.13.1 or higher

# Install Requirement
pip install -r requirements.txt

# Setup
Install the required packages by running pip install -r requirements.txt
Set the ASSEMBLYAI_API_KEY environment variable to your AssemblyAI API key
Run the application with python app.py

# Usage
Navigate to http://localhost:8800 in your web browser
Click "Choose File" to select an audio file to upload (supported formats: WAV, MP3, FLAC)
Click "Upload" to submit the file for transcription and summarization
Once the file has been processed, you will be redirected to a page displaying the transcript and summary

# Security and Upload Guardrails
- API key is loaded from environment variable: `ASSEMBLYAI_API_KEY`
- Uploads are rate-limited per IP (5 uploads per 60 seconds)
- Uploaded files are stored with UUID filenames to avoid filename collisions
- Runtime files are stored in `data/uploads/` (not committed to git)

# History UI
- `/history` shows all jobs with search and status filter (`all`, `done`, `failed`)
- Each history row has quick actions: view result, download transcript/summary/SRT/VTT

# Code Structure
app.py: the main Flask application file
index(): the home page
upload_file(): handles file uploads, transcription, and summarization
uploaded_file(): displays the transcript and summary for a given file
templates/index.html: the home page template
templates/uploaded_file.html: the template for displaying transcripts and summaries
data/uploads/: runtime upload files and `jobs_history.json`

# Transcription and Summarization
Transcription is performed using the AssemblyAI API, and summarization is performed using the Hugging Face Transformers library.
