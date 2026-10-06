import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import tkinter as tk
import os
import threading
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import Cerebrum_v0


SAMPLE_RATE = 16000
SECONDS = 5

INITIAL_FOLDER = "C:/Users/tiger/Desktop/Python/Cerebrum/projects/samples"
DATASET_FOLDER = "C:/Users/tiger/Desktop/Python/Cerebrum/projects/dataset"


# ---------- Recording ----------

def record():
    name = entryname.get().strip()

    if name == "":
        name = "nothing"

    folder = entryfile.get().strip()

    if folder == "":
        folder = INITIAL_FOLDER

    os.makedirs(folder, exist_ok=True)

    thread = threading.Thread(
        target=record_audio,
        args=(name, folder),
        daemon=True
    )

    thread.start()


def record_audio(name, folder):
    audio = sd.rec(
        int(SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    root.after(0, update_timer, SECONDS)

    sd.wait()

    path = os.path.join(folder, f"{name}.wav")

    sf.write(path, audio, SAMPLE_RATE)

    root.after(0, recording_finished)
    root.after(0, update_file_list)


def update_timer(seconds):
    if seconds > 0:
        timerlabel.config(text=f"{seconds} s")
        root.after(1000, update_timer, seconds - 1)


def recording_finished():
    timerlabel.config(text="Done!")


# ---------- File list ----------

def update_file_list():
    folder = entryfile.get().strip()

    if folder == "":
        folder = INITIAL_FOLDER

    if not os.path.exists(folder):
        return

    listbox.delete(0, tk.END)

    files = os.listdir(folder)

    for file in sorted(files):
        if file.lower().endswith(".wav"):
            listbox.insert(tk.END, file)


# ---------- Audio playback ----------

def play_audio():
    selection = listbox.curselection()

    if not selection:
        return

    filename = listbox.get(selection[0])

    folder = entryfile.get().strip()

    if folder == "":
        folder = INITIAL_FOLDER

    path = os.path.join(folder, filename)

    audio, sr = sf.read(path)

    sd.play(audio, sr)


def stop_audio():
    sd.stop()


# ---------- Dataset ----------

def get_next_dataset_number():
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    numbers = []

    for file in os.listdir(DATASET_FOLDER):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]

            if name.isdigit():
                numbers.append(int(name))

    if len(numbers) == 0:
        return 0

    return max(numbers) + 1


def add_to_dataset():
    selection = listbox.curselection()

    if not selection:
        dataset_status.config(text="Select a WAV file")
        return

    label = entrylabel.get().strip()

    if label == "":
        dataset_status.config(text="Enter a label")
        return

    filename = listbox.get(selection[0])

    folder = entryfile.get().strip()

    if folder == "":
        folder = INITIAL_FOLDER

    wav_path = os.path.join(folder, filename)

    try:
        audio, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13
        )

        data = {
            "x": mfcc,
            "label": label
        }

        number = get_next_dataset_number()

        dataset_path = os.path.join(
            DATASET_FOLDER,
            f"{number}.npy"
        )

        np.save(
            dataset_path,
            data,
            allow_pickle=True
        )

        dataset_status.config(
            text=f"Added: {number}.npy"
        )

    except Exception as e:
        dataset_status.config(
            text=f"Error: {e}"
        )


# ---------- GUI ----------

root = tk.Tk()
root.title("Speech Dataset")
root.geometry("700x500")


# WAV name

tk.Label(
    root,
    text="Recording name:"
).pack(anchor="w", padx=10, pady=(10, 0))

entryname = tk.Entry(root)
entryname.pack(fill="x", padx=10)

entryname.insert(0, "nothing")


# WAV folder

tk.Label(
    root,
    text="WAV folder:"
).pack(anchor="w", padx=10, pady=(10, 0))

entryfile = tk.Entry(root)
entryfile.pack(fill="x", padx=10)

entryfile.insert(0, INITIAL_FOLDER)


# Recording controls

record_frame = tk.Frame(root)
record_frame.pack(pady=10)

recordbutton = tk.Button(
    record_frame,
    text="Record",
    command=record
)

recordbutton.pack(side="left", padx=5)

timerlabel = tk.Label(
    record_frame,
    text="Ready"
)

timerlabel.pack(side="left", padx=10)


# File list

tk.Label(
    root,
    text="WAV files:"
).pack(anchor="w", padx=10)

listbox = tk.Listbox(root)
listbox.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=5
)


# Playback controls

play_frame = tk.Frame(root)
play_frame.pack(pady=5)

playbutton = tk.Button(
    play_frame,
    text="Play",
    command=play_audio
)

playbutton.pack(side="left", padx=5)

stopbutton = tk.Button(
    play_frame,
    text="Stop",
    command=stop_audio
)

stopbutton.pack(side="left", padx=5)


# Dataset label

tk.Label(
    root,
    text="Label:"
).pack(anchor="w", padx=10, pady=(10, 0))

entrylabel = tk.Entry(root)
entrylabel.pack(fill="x", padx=10)

entrylabel.insert(0, "nothing")


# Dataset button

datasetbutton = tk.Button(
    root,
    text="Add to dataset",
    command=add_to_dataset
)

datasetbutton.pack(pady=10)


dataset_status = tk.Label(
    root,
    text="Ready"
)

dataset_status.pack()


# Load files

update_file_list()


root.mainloop()