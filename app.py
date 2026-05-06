import os
import re
import shutil
import unicodedata
from difflib import SequenceMatcher
from typing import List, Optional

import whisper
import pykakasi
from Levenshtein import ratio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


def _normalize_segment_hira(hira: str) -> str:
    return _PUNCT_RE.sub("", unicodedata.normalize("NFKC", hira or "")).lower()


def _build_token_matches(target_text: str, user_norm: str):
    """
    Tokenize target_text via pykakasi and align it against user_norm
    (already-normalized hiragana of what the user actually said).
    Returns [{orig, hira, matched}] in original order.
    A token is "matched" if at least half of its normalized hiragana
    characters fall inside an "equal" block from SequenceMatcher.
    """
    target_segments = kks.convert(target_text or "")

    target_norm_full = ""
    seg_ranges = []
    for seg in target_segments:
        seg_norm = _normalize_segment_hira(seg.get("hira", ""))
        start = len(target_norm_full)
        target_norm_full += seg_norm
        end = len(target_norm_full)
        seg_ranges.append((seg, start, end))

    matched = [False] * len(target_norm_full)
    if target_norm_full and user_norm:
        matcher = SequenceMatcher(None, target_norm_full, user_norm, autojunk=False)
        for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
            if tag == "equal":
                for k in range(i1, i2):
                    matched[k] = True

    tokens = []
    for seg, start, end in seg_ranges:
        if start == end:
            is_matched = True  # punctuation/whitespace, never fade
        else:
            hits = sum(1 for k in range(start, end) if matched[k])
            is_matched = (hits / (end - start)) >= 0.5
        tokens.append({
            "orig": seg.get("orig", ""),
            "hira": seg.get("hira", ""),
            "matched": is_matched,
        })
    return tokens


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
    tokens = _build_token_matches(target_text, user_norm)

    return {
        "target": target_text,
        "user_said": user_text,
        "score": round(score, 2),
        "target_hira": target_norm,
        "user_hira": user_norm,
        "tokens": tokens,
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


# ===== Furigana =====

class FuriganaRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None


def _segment_text(text: str):
    if not text:
        return []
    return [
        {"orig": item.get("orig", ""), "hira": item.get("hira", "")}
        for item in kks.convert(text)
    ]


@app.post("/furigana")
def furigana(req: FuriganaRequest):
    if req.texts is not None:
        return {"results": [_segment_text(t) for t in req.texts]}
    return {"segments": _segment_text(req.text or "")}
