import whisper
from transformers import MarianMTModel, MarianTokenizer
import subprocess
import os

def transcribe(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result["text"]

def translate(text, target_lang="ar"):
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer(text, return_tensors="pt"))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def text_to_speech(text, output_path="dubbed_audio.wav"):
    command = f'mimic3 --voice ar_JO "{text}" --output-dir {output_path}'
    subprocess.run(command, shell=True)
    return os.path.join(output_path, "dubbed_audio.wav")

def merge_audio_video(video_path, audio_path, output_path="output.mp4"):
    command = f'ffmpeg -i {video_path} -i {audio_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {output_path}'
    subprocess.run(command, shell=True)
    return output_path

def dub_video(video_path, lang):
    # الخطوة 1: تحويل الصوت إلى نص
    text = transcribe(video_path)
    
    # الخطوة 2: ترجمة النص
    translated_text = translate(text, lang)
    
    # الخطوة 3: توليد الصوت
    audio_path = text_to_speech(translated_text)
    
    # الخطوة 4: دمج الصوت مع الفيديو
    output_path = merge_audio_video(video_path, audio_path)
    
    return output_path
  
