import whisper
import pykakasi
from Levenshtein import ratio
from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os

app = FastAPI()

model = whisper.load_model("base")
kks = pykakasi.kakasi()

def to_hiragana(text):
    result = kks.convert(text)
    return "".join([item['hira'] for item in result])

def assess_pronunciation(audio_path, target_text):
    result = model.transcribe(
        audio_path,
        language="ja",
        task="transcribe",
        temperature=0
    )

    user_text = result['text'].strip()

    target_hira = to_hiragana(target_text)
    user_hira = to_hiragana(user_text)

    score = ratio(target_hira, user_hira) * 100

    return {
        "target": target_text,
        "user_said": user_text,
        "score": round(score, 2)
    }

@app.post("/assess")
async def assess(
    file: UploadFile = File(...),
    target_text: str = Form(...)
):
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = assess_pronunciation(temp_path, target_text)
    finally:
        os.remove(temp_path)  

    return result