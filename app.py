import os
import re
import shutil
import unicodedata

import whisper
import pykakasi
from Levenshtein import ratio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# "small" cho độ chính xác tốt hơn nhiều với tiếng Nhật so với "tiny".
# Nếu HF Space đủ RAM/CPU, có thể nâng lên "medium".
MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
model = whisper.load_model(MODEL_NAME)
kks = pykakasi.kakasi()

# Bỏ dấu câu/khoảng trắng tiếng Nhật + ASCII trước khi so sánh
_PUNCT_RE = re.compile(
    r"[\s　、。，．！？・"
    r"「」『』【】（）"
    r".,!?\"'`~\-_/\\(){}\[\]<>:;]"
)


def to_hiragana(text: str) -> str:
    result = kks.convert(text)
    return "".join(item["hira"] for item in result)


def normalize_for_compare(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    hira = to_hiragana(text)
    return _PUNCT_RE.sub("", hira).lower()


def assess_pronunciation(audio_path: str, target_text: str) -> dict:
    # initial_prompt giúp Whisper "biết trước" văn bản kỳ vọng
    # → giảm rất nhiều lỗi nhận dạng âm gần giống.
    # Whisper giới hạn prompt ~224 tokens, cắt cho an toàn.
    prompt = target_text.strip()[:200]

    result = model.transcribe(
        audio_path,
        language="ja",
        task="transcribe",
        temperature=0.0,
        beam_size=5,
        best_of=5,
        initial_prompt=prompt,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
    )

    user_text = result["text"].strip()

    target_norm = normalize_for_compare(target_text)
    user_norm = normalize_for_compare(user_text)

    score = ratio(target_norm, user_norm) * 100

    return {
        "target": target_text,
        "user_said": user_text,
        "score": round(score, 2),
        "target_hira": target_norm,
        "user_hira": user_norm,
    }


@app.post("/assess")
async def assess(
    file: UploadFile = File(...),
    target_text: str = Form(...),
):
    safe_name = os.path.basename(file.filename or "audio.bin")
    temp_path = f"temp_{safe_name}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        return assess_pronunciation(temp_path, target_text)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
