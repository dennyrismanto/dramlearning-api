from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse
import whisper  # Whisper library for speech recognition
import json
import os
import pyttsx3
import logging
import shutil
import sys
import time
from pathlib import Path
import hashlib
import requests
from tqdm import tqdm
import google.generativeai as genai
import uuid
from datetime import datetime
import re
from transformers import AutoModelForCausalLM, AutoTokenizer  # New imports

try:
    from dotenv import load_dotenv
except ImportError:
    print("""
Error: python-dotenv package is not installed.
Please install it using pip:

    pip install python-dotenv

Then run this application again.
""")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_ffmpeg():
    """Check if ffmpeg is installed and provide installation instructions if not."""
    if shutil.which("ffmpeg") is None:
        logger.error("""
FFmpeg tidak ditemukan! Anda perlu menginstall FFmpeg:

1. Download FFmpeg dari: https://github.com/BtbN/FFmpeg-Builds/releases
   - Pilih ffmpeg-master-latest-win64-gpl.zip
   
2. Ekstrak file zip yang sudah didownload

3. Tambahkan path folder bin dari FFmpeg ke System Environment Variables:
   - Buka Control Panel -> System -> Advanced System Settings
   - Klik Environment Variables
   - Di System Variables, cari PATH
   - Klik Edit dan Add
   - Masukkan path folder bin FFmpeg (contoh: C:\\ffmpeg\\bin)
   - Klik OK di semua jendela
   
4. Restart terminal/command prompt dan aplikasi

5. Untuk verifikasi, jalankan: ffmpeg -version
""")
        sys.exit(1)
    else:
        logger.info("FFmpeg terdeteksi dan siap digunakan")

# Check FFmpeg before starting the application
check_ffmpeg()

app = FastAPI()

# Create audio_files directory if it doesn't exist
os.makedirs("audio_files", exist_ok=True)

# Allowed audio formats
ALLOWED_AUDIO_FORMATS = ['.m4a', '.mp3', '.wav']

def download_with_progress(url, destination):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def calculate_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def is_file_locked(filepath):
    """Check if a file is locked/being used by another process"""
    try:
        with open(filepath, 'a') as _:
            pass
        return False
    except IOError:
        return True

def wait_for_file_unlock(filepath, timeout=30, check_interval=1):
    """Wait for a file to become unlocked with timeout"""
    start_time = time.time()
    while is_file_locked(filepath):
        if time.time() - start_time > timeout:
            return False
        time.sleep(check_interval)
    return True

def safe_remove_file(file_path, max_attempts=5, delay=1):
    """Safely remove a file with retries for locked files"""
    if not os.path.exists(file_path):
        return True
        
    # First wait for file to become unlocked
    if not wait_for_file_unlock(file_path):
        logger.error(f"File {file_path} is still locked after timeout")
        return False
        
    for i in range(max_attempts):
        try:
            os.remove(file_path)
            return True
        except PermissionError:
            if i < max_attempts - 1:
                logger.info(f"File {file_path} is locked, waiting {delay} seconds before retry {i+1}/{max_attempts}")
                time.sleep(delay)
            continue
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {str(e)}")
            return False
    return False

def initialize_whisper_model():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "tiny.pt"  # Using tiny model which is much smaller and more stable
    temp_model_path = model_dir / "tiny.pt.downloading"
    
    # Model URL and expected checksum for tiny model
    model_url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
    expected_checksum = "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9"
    
    max_retries = 3
    attempt = 0
    
    while attempt < max_retries:
        try:
            # If model exists, verify checksum
            if model_path.exists():
                try:
                    # Wait for file to become available
                    if not wait_for_file_unlock(str(model_path)):
                        raise Exception("Model file is locked and timeout reached")
                        
                    current_hash = calculate_file_hash(model_path)
                    if current_hash == expected_checksum:
                        logger.info("Existing model file verified successfully")
                        return whisper.load_model("tiny", download_root=str(model_dir))
                except Exception as e:
                    logger.warning(f"Error verifying existing model: {str(e)}")
                
                # If we get here, either verification failed or there was an error
                if not safe_remove_file(str(model_path)):
                    logger.error("Could not remove existing model file - it may be in use")
                    raise Exception("Could not remove existing model file - it may be in use")
            
            # Clean up any existing temporary files
            safe_remove_file(str(temp_model_path))
            
            # Download model to temporary file
            logger.info("Downloading Whisper model...")
            download_with_progress(model_url, temp_model_path)
            
            # Verify downloaded file
            downloaded_hash = calculate_file_hash(temp_model_path)
            if downloaded_hash != expected_checksum:
                safe_remove_file(str(temp_model_path))
                raise Exception(f"Downloaded model checksum verification failed. Expected: {expected_checksum}, Got: {downloaded_hash}")
            
            # If verification successful, move temp file to final location
            if model_path.exists():
                if not safe_remove_file(str(model_path)):
                    safe_remove_file(str(temp_model_path))
                    raise Exception("Could not remove existing model file - it may be in use")
                    
            os.rename(temp_model_path, model_path)
            
            logger.info("Model downloaded and verified successfully")
            return whisper.load_model("tiny", download_root=str(model_dir))
            
        except Exception as e:
            attempt += 1
            logger.error(f"Attempt {attempt} failed: {str(e)}")
            
            # Clean up any temporary files
            safe_remove_file(str(temp_model_path))
            
            if attempt >= max_retries:
                logger.error("Maximum retries reached. Could not load Whisper model.")
                raise
            
            logger.info(f"Retrying... (attempt {attempt + 1}/{max_retries})")
            time.sleep(5)  # Add delay between retries

# Initialize offline model
def initialize_offline_model():
    try:
        model_name = "microsoft/phi-2"  # Using a smaller model that works well offline
        logger.info(f"Loading offline model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        
        logger.info("Offline model initialized successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error initializing offline model: {str(e)}")
        raise

def init_gemini():
    """Initialize Gemini API"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')  # Changed from 'gemini-pro' to 'gemini-1.0-pro'
        logger.info("Gemini API initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Error initializing Gemini: {str(e)}")
        raise

def generate_response(prompt):
    """Generate response using Gemini API"""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return "Maaf, terjadi kesalahan dalam pemrosesan. Silakan coba lagi nanti."

# Initialize models
try:
    whisper_model = initialize_whisper_model()
    offline_model, offline_tokenizer = initialize_offline_model()  # Initialize offline model
    model = init_gemini()
    logger.info("Models initialized successfully")
except Exception as e:
    logger.error(f"Critical error initializing models: {str(e)}")
    raise

# Load knowledge base
try:
    with open("knowledge_base.json", "r", encoding='utf-8') as f:
        knowledge_base = json.load(f)
    logger.info("Knowledge base loaded successfully")
except FileNotFoundError:
    logger.warning("Knowledge base file not found. Creating empty knowledge base")
    knowledge_base = {}
except json.JSONDecodeError:
    logger.error("Invalid JSON in knowledge base file")
    raise

# TTS engine setup
try:
    engine = pyttsx3.init()
    # Get all available voices
    voices = engine.getProperty('voices')
    
    # Find Indonesian voice
    indonesian_voice = None
    for voice in voices:
        if "indonesian" in voice.name.lower() or "charon" in voice.name.lower():
            indonesian_voice = voice
            break
    
    if indonesian_voice:
        engine.setProperty('voice', indonesian_voice.id)
    else:
        logger.warning("Indonesian voice not found, using default voice")
        engine.setProperty('voice', voices[0].id)
        
    # Configure voice settings
    engine.setProperty('rate', 150)    # Speaking rate
    engine.setProperty('volume', 0.9)  # Volume (0-1)
    logger.info(f"TTS engine initialized with voice: {engine.getProperty('voice')}")
except Exception as e:
    logger.error(f"Error initializing TTS engine: {str(e)}")
    raise

def clean_text(text):
    """Clean text from special characters and normalize spacing"""
    # Remove asterisks, quotes, and normalize newlines
    cleaned = re.sub(r'[\*\"\'\n]+', ' ', text)
    # Normalize spaces (remove multiple spaces)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    return cleaned

def tts_speak_to_file(text, filename=None):
    try:
        # Clean the text before TTS processing
        cleaned_text = clean_text(text)
        
        if filename is None:
            # Generate unique filename with date and random ID
            date_str = datetime.now().strftime("%Y%m%d")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"audio_files/{date_str}_response_{unique_id}.wav"
        
        engine.save_to_file(cleaned_text, filename)
        engine.runAndWait()
        return filename
    except Exception as e:
        logger.error(f"Error in TTS conversion: {str(e)}")
        raise HTTPException(status_code=500, detail="TTS conversion failed")

@app.post("/voicebot")
async def voicebot(audio: UploadFile = File(...)):
    try:
        # Validate file extension
        file_extension = os.path.splitext(audio.filename)[1].lower()
        if file_extension not in ALLOWED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format. Allowed formats: {', '.join(ALLOWED_AUDIO_FORMATS)}"
            )

        # 1. Save uploaded audio
        audio_path = os.path.join("audio_files", audio.filename)
        try:
            with open(audio_path, "wb") as f:
                content = await audio.read()
                if not content:
                    raise HTTPException(status_code=400, detail="Empty audio file")
                f.write(content)
            
            logger.info(f"Audio saved successfully to: {audio_path}")
            logger.info(f"Audio file size: {os.path.getsize(audio_path)} bytes")
            
        except Exception as e:
            logger.error(f"Error saving audio file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving audio file: {str(e)}")
        
        # 2. Transcribe
        try:
            logger.info("Starting transcription with Whisper...")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found at {audio_path}")
                
            result = whisper_model.transcribe(audio_path)
            if not result or "text" not in result:
                raise ValueError("Transcription result is empty or invalid")
                
            text = result["text"]
            logger.info(f"Transcription successful. Text: {text}")
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Audio file not found: {str(e)}")
        except Exception as e:
            logger.error(f"Transcription error: {type(e).__name__} - {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

        # 3. Check Knowledge Base
        response_text = None
        for key in knowledge_base:
            if key.lower() in text.lower():
                response_text = knowledge_base[key]
                break

        # 4. If no match, use Gemini API
        if not response_text:
            response_text = generate_response(text)

        # 5. TTS: convert to voice
        audio_output_path = tts_speak_to_file(response_text)

        return {
            "question": text,
            "answer": response_text,
            "audio_url": audio_output_path
        }
    except Exception as e:
        logger.error(f"Error in voicebot endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/textbot", 
    response_model=dict,
    summary="Kirim pesan teks ke bot",
    description="""
    Kirim pertanyaan teks ke bot dan terima respons teks dan audio
    Bot akan mencoba mencocokkan pertanyaan dengan basis pengetahuan terlebih dahulu
    Jika tidak ada yang cocok bot akan menghasilkan respons menggunakan model AI
    """,
    responses={
        200: {
            "description": "Respons berhasil",
            "content": {
                "application/json": {
                    "example": {
                        "question": "apa kabar",
                        "answer": "baik terimakasih ada yang bisa saya bantu",
                        "audio_url": "audio_files/response.wav"
                    }
                }
            }
        },
        422: {"description": "Error Validasi"},
        500: {"description": "Error Server Internal"}
    }
)
async def textbot(question: str = Body(..., embed=True, description="Teks pertanyaan yang akan dikirim ke bot")):
    try:
        if not question or question.strip() == "":
            raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong")

        # Check Knowledge Base
        response_text = None
        for key in knowledge_base:
            if key.lower() in question.lower():
                response_text = knowledge_base[key]
                break

        # If no match, use Gemini API
        if not response_text:
            response_text = generate_response(question)

        # Generate audio response
        try:
            audio_output_path = tts_speak_to_file(response_text)
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error generating audio response")

        return {
            "question": question,
            "answer": response_text,
            "audio_url": audio_output_path
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in textbot endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")

@app.post("/textbot/offline",
    response_model=dict,
    summary="Send text message to bot using offline model",
    description="""
    Send a text question to the bot and receive a text response using the offline model.
    This endpoint works completely offline without requiring internet connection.
    """,
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "question": "apa kabar",
                        "answer": "baik terimakasih ada yang bisa saya bantu",
                        "audio_url": "audio_files/response.wav"
                    }
                }
            }
        },
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)
async def textbot_offline(question: str = Body(..., embed=True, description="Question text to send to the bot")):
    try:
        if not question or question.strip() == "":
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Check Knowledge Base first
        response_text = None
        for key in knowledge_base:
            if key.lower() in question.lower():
                response_text = knowledge_base[key]
                break

        # If no match in knowledge base, use offline model
        if not response_text:
            # Format the prompt for the model
            prompt = f"Q: {question}\nA:"
            
            # Generate response using offline model
            inputs = offline_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = offline_model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=offline_tokenizer.eos_token_id
            )
            response_text = offline_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response by removing the prompt and any extra whitespace
            response_text = response_text.replace(prompt, "").strip()

        # Generate audio response
        try:
            audio_output_path = tts_speak_to_file(response_text)
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error generating audio response")

        return {
            "question": question,
            "answer": response_text,
            "audio_url": audio_output_path
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in textbot_offline endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = f"audio_files/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
