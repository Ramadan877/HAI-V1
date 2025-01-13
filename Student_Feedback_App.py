from flask import Flask, request, render_template, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import openai
import os
from gtts import gTTS
import whisper  
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

openai.api_key = "OpenAiApiKey"

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads/'
USER_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'user_audio')
AI_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'ai_audio')
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg', 'gif', 'pdf', 'mp4', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['USER_AUDIO_FOLDER'] = USER_AUDIO_FOLDER
app.config['AI_AUDIO_FOLDER'] = AI_AUDIO_FOLDER

os.makedirs(USER_AUDIO_FOLDER, exist_ok=True)
os.makedirs(AI_AUDIO_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = whisper.load_model("small")

def speech_to_text(audio_file_path):
    """Convert audio to text using Whisper."""
    result = model.transcribe(audio_file_path)
    return result["text"]

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/resources/<path:filename>')
def download_resource(filename):
    """Serve resources like PDF or video files from the resources folder."""
    return send_from_directory('resources', filename)


@app.route('/set_context', methods=['POST'])
def set_context():
    """Set the context for the explanation (e.g., based on a PDF or video)."""
    resource_type = request.form.get('type')  
    description = request.form.get('The video which the student was provided is explaining how to build a face recognition app using Python')  # Textual description of the resource
    golden_answer = request.form.get('using the openCV Python package to load a pre-trained  Haar cascade classifier which is used to detect faces in video frames, next to access the default camera on the laptop to capture video frames continuously, each frame is then converted into grayscale and then the classifier detects the faces. If any are found, it draws rectangles around them, the detected faces are then displayed on the outputted video footage on the screen.')  # Correct explanation

    if not description or not golden_answer:
        return jsonify({'error': 'Description and golden answer are required to set context.'})

    if resource_type not in ['pdf', 'video']:
        return jsonify({'error': 'Invalid resource type.'})

    session['original_text'] = description
    session['resource_type'] = resource_type
    session['golden_answer'] = golden_answer
    session['attempt_count'] = 0

    return jsonify({'message': f'Context set for {resource_type} file.'})

@app.route('/submit_message', methods=['POST'])
def submit_message():
    """Handle the submission of user messages and generate AI responses."""
    user_message = request.form.get('message')
    audio_file = request.files.get('audio')

    if not user_message and not audio_file:
        return jsonify({'error': 'Message or audio is required.'})

    user_audio_url = None

    if audio_file:
        audio_filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['USER_AUDIO_FOLDER'], audio_filename)
        audio_file.save(audio_path)

        try:
            audio = AudioSegment.from_file(audio_path)
            wav_path = os.path.splitext(audio_path)[0] + '.wav'
            audio.export(wav_path, format='wav')

            user_message = speech_to_text(wav_path)

            user_audio_url = f"/uploads/user_audio/{audio_filename}"
        except Exception as e:
            return jsonify({'error': f'Audio processing error: {str(e)}'})
    elif not user_message:
        return jsonify({'error': 'No audio or text message provided.'})

    original_text = session.get('original_text', 'No context provided.')
    resource_type = session.get('resource_type', 'text')
    golden_answer = session.get('golden_answer', 'Correct explanation of the concept.')
    attempt_count = session.get('attempt_count', 0)

    attempt_count += 1
    session['attempt_count'] = attempt_count

    ai_response = generate_response(user_message, original_text, resource_type, golden_answer, attempt_count)

    if attempt_count >= 3:
        session.pop('attempt_count', None)  

    ai_response_filename = 'response_audio.mp3'
    audio_response_path = os.path.join(app.config['AI_AUDIO_FOLDER'], ai_response_filename)
    generate_audio(ai_response, audio_response_path)

    ai_audio_url = f"/uploads/ai_audio/{ai_response_filename}"

    return jsonify({
        'response': ai_response,
        'transcription': user_message,
        'user_audio_url': user_audio_url,
        'ai_audio_url': ai_audio_url
    })


def generate_response(user_message, original_text, resource_type, golden_answer, attempt_count):
    """Generate a response dynamically using OpenAI GPT."""

    if not golden_answer or not original_text:
        return "As your tutor, I'm not able to provide you with feedback without having context about your explanation. Please ensure the context is set."
    
    base_prompt = f"""
    Context: {original_text}
    Resource Type: {resource_type}
    Golden Answer: {golden_answer}
    User Explanation: {user_message}
    
    You are a tutor, Your task is to evaluate the student's explanation and provide feedback:
    - If the user's explanation is very accurate, acknowledge it.
    - If the explanation is partially correct, point out the key areas that need improvement.
    - If it's incorrect, provide constructive feedback and suggest improvements.
    - Offer increasingly specific hints or the correct answer after multiple attempts.
    """

    if attempt_count == 1:
        base_prompt += "\nProvide general feedback and a broad hint to guide the user."
    elif attempt_count == 2:
        base_prompt += "\nProvide more specific feedback and highlight key elements the user missed."
    elif attempt_count == 3:
        base_prompt += "\nProvide the correct explanation, as the user has made multiple attempts."
    else:
        base_prompt += "\nReset and encourage the user to try again with a fresh start."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful tutor providing feedback to students."},
                {"role": "user", "content": base_prompt}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        ai_response = response['choices'][0]['message']['content']
        return ai_response
    except Exception as e:
        return f"Error generating AI response: {str(e)}"


def compare_explanation(user_message, golden_answer):
    """Compare user explanation with the golden answer and provide qualitative feedback."""
    if user_message.lower() in golden_answer.lower():
        return "very accurate"
    elif some_partial_match(user_message, golden_answer): 
        return "partially correct"
    else:
        return "not accurate"

def some_partial_match(user_message, golden_answer):
    """Placeholder for partial matching logic."""
    return False

def generate_audio(text, file_path):
    """Generate speech (audio) from the provided text using gTTS."""
    tts = gTTS(text=text, lang='en')
    tts.save(file_path)

@app.route('/uploads/<folder>/<filename>')
def serve_audio(folder, filename):
    """Serve the audio files from the uploads folder."""
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    return send_from_directory(folder_path, filename)

if __name__ == '__main__':
    app.run(debug=True)
