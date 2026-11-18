import sounddevice as sd
import numpy as np
import webrtcvad
import queue
import threading
import re

from faster_whisper import WhisperModel
from collections import deque
from difflib import SequenceMatcher

# =========================
# Налаштування
# =========================

SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

SILENCE_LIMIT = 500
VAD_MODE = 2

LANGUAGE = "en"
SIMILARITY = 0.75

# =========================
# Спели
# =========================

SPELLS = [
    "fireball",
    "freeze",
    "divine light",
    "lightning",
    "heal",
    "shield",
]

# =========================
# Модель
# =========================

print("Завантаження моделі...")

model = WhisperModel("tiny", device="cpu", compute_type="int8")

vad = webrtcvad.Vad(VAD_MODE)

print("start")


# Черга для передачі аудіо іншому потоку
audio_queue = queue.Queue()

# =========================
# Пошук спела
# =========================


def find_spell(text):

    text = text.lower()
    words = re.findall(r"[a-z]+", text)

    best_spell = None
    best_score = 0

    for spell in SPELLS:

        spell_words = spell.split()
        count = len(spell_words)

        # Перевіряємо послідовності слів
        for i in range(len(words) - count + 1):

            phrase = " ".join(words[i : i + count])

            score = SequenceMatcher(None, phrase, spell).ratio()

            if score > best_score:
                best_score = score
                best_spell = spell

    if best_score >= SIMILARITY:
        return best_spell

    return None


# =========================
# guess
# =========================


def recognize(audio):

    audio = np.concatenate(audio).flatten().astype(np.float32)

    segments, info = model.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    text = " ".join(segment.text.strip() for segment in segments).lower()

    # text
    print("Recognised:", text)

    spell = find_spell(text)

    if spell:
        print(f"CAST: {spell}\n")


# =========================
# thread Whisper
# =========================


def recognition_worker():

    while True:

        audio = audio_queue.get()

        try:
            recognize(audio)

        except Exception as e:
            print("Error of recognition:", e)

        finally:
            audio_queue.task_done()


threading.Thread(target=recognition_worker, daemon=True).start()


# =========================
# listening
# =========================

audio_buffer = []
pre_buffer = deque(maxlen=10)

is_speaking = False
silence_frames = 0


def callback(indata, frames, time, status):

    global is_speaking
    global silence_frames
    global audio_buffer

    # float32 -> int16 для WebRTC VAD
    pcm = np.clip(indata[:, 0] * 32768, -32768, 32767).astype(np.int16)

    speech = vad.is_speech(pcm.tobytes(), SAMPLE_RATE)

    frame = indata.copy()

    if not is_speaking:
        pre_buffer.append(frame)

    if speech:

        if not is_speaking:
            is_speaking = True
            audio_buffer = list(pre_buffer)
            pre_buffer.clear()

        silence_frames = 0
        audio_buffer.append(frame)

    elif is_speaking:

        audio_buffer.append(frame)
        silence_frames += 1

        if silence_frames * FRAME_DURATION >= SILENCE_LIMIT:

            finished_audio = audio_buffer.copy()

            audio_buffer = []
            silence_frames = 0
            is_speaking = False

            # Передаємо аудіо в чергу
            # whisper його обробить в іншому потоці
            audio_queue.put(finished_audio)


# =========================
# Start
# =========================

try:

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE,
        callback=callback,
    ):

        while True:
            sd.sleep(100)

except KeyboardInterrupt:
    print("\nStopped.")
