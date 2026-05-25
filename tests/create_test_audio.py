"""
Generate a mixed Hindi / Tamil / Telugu WAV file for SarvamLID testing.
Requires: gtts, pydub, ffmpeg

Usage:
    python3 tests/create_test_audio.py [output_path]
    # default output: tests/mixed_lid_test.wav
"""

import sys
import os
import io
import tempfile
from gtts import gTTS
from pydub import AudioSegment

SEGMENTS = [
    ("hi", "नमस्ते, मेरा नाम राम है। मैं दिल्ली से हूँ। आज मौसम बहुत अच्छा है।"),
    ("ta", "வணக்கம், என் பெயர் குமார். நான் சென்னையில் இருந்து வருகிறேன். இன்று வானிலை மிகவும் நல்லாக இருக்கிறது."),
    ("te", "నమస్కారం, నా పేరు రాజు. నేను హైదరాబాద్ నుండి వచ్చాను. ఈరోజు వాతావరణం చాలా బాగుంది."),
    ("hi", "क्या आप मुझे बता सकते हैं कि यहाँ का सबसे अच्छा रेस्टोरेंट कौन सा है?"),
    ("ta", "நீங்கள் எங்கே வசிக்கிறீர்கள்? நான் உங்களுக்கு உதவ விரும்புகிறேன்."),
    ("te", "మీరు ఏమి చేస్తున్నారు? నేను మీకు సహాయం చేయగలను."),
]

SILENCE_MS = 800  # pause between segments


def make_segment(lang: str, text: str) -> AudioSegment:
    tts = gTTS(text=text, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    audio = AudioSegment.from_mp3(buf)
    return audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "tests/mixed_lid_test.wav"

    silence = AudioSegment.silent(duration=SILENCE_MS, frame_rate=16000)
    combined = AudioSegment.silent(duration=500, frame_rate=16000)  # leading silence

    for i, (lang, text) in enumerate(SEGMENTS):
        print(f"  Generating segment {i + 1}/{len(SEGMENTS)}: {lang} — {text[:40]}...")
        seg = make_segment(lang, text)
        combined += seg + silence

    combined += AudioSegment.silent(duration=500, frame_rate=16000)  # trailing silence

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    combined.export(out_path, format="wav")
    duration = len(combined) / 1000
    print(f"\nSaved: {out_path} ({duration:.1f}s, mono 16kHz 16-bit)")
    return out_path


if __name__ == "__main__":
    main()
