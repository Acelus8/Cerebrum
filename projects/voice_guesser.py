import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import tkinter as tk
import threading
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))
import Cerebrum_v0

SAMPLE_RATE = 16000
SECONDS = 5
AI_FOLDER = "C:/Users/tiger/Desktop/Python/Cerebrum/projects/ai/voice_ai.npz"
DATASET_FOLDER = "C:/Users/tiger/Desktop/Python/Cerebrum/projects/dataset"

Voice_Ai = Cerebrum_v0.AI(2041, [64, 64, 32, 32, 4])
# Voice_Ai.load(AI_FOLDER)


d = {0: "nothing", 1: "fireball", 2: "freeze", 3: "divine light"}
d_reverse = {value: key for key, value in d.items()}
d_ai = {v: np.eye(len(d))[k] for k, v in d.items()}

class Guesser:
    def __init__(self):
        self.ai = Voice_Ai

    def guess(self, audio):
        guess = self.ai.calculate(audio)

        g = np.max(guess)
        index = np.where(guess == g)[0][0]
        print()
        print(d[index])
        print()
        
        print("nothing:", guess[0])
        print("fireball:", guess[1])
        print("freeze:", guess[2])
        print("divine_light:", guess[3])
        
        
        
        
        

    def learn(self, times=10000):
        files = [
            file for file in os.listdir(DATASET_FOLDER)
            if file.endswith(".npy")
        ]

        for i in range(times):
            np.random.shuffle(files)

            for file in files:
                path = os.path.join(DATASET_FOLDER, file)
                data = np.load(path, allow_pickle=True).item()

                x = data["x"]
                label = data["label"]

                self.ai.learn(x, d_ai[label])

            print(f"Epoch {i + 1}/{times}")

        print("finished, sir")

    def save(self):
        self.ai.save(AI_FOLDER)
        print("saved, sir")
    
    
guesser = Guesser()


# ---------- Recording ----------


def record():

    thread = threading.Thread(target=record_audio, args=(), daemon=True)

    thread.start()


def record_audio():
    audio = sd.rec(
        int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    )
    
    root.after(0, update_timer, SECONDS)

    sd.wait()

    root.after(0, recording_finished)
    audio = audio.flatten()
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    mfcc = mfcc.flatten()
    guesser.guess(mfcc)

def update_timer(seconds):
    if seconds > 0:
        timerlabel.config(text=f"{seconds} s")
        root.after(1000, update_timer, seconds - 1)


def recording_finished():
    timerlabel.config(text="Done!")


#  ---------- GUI


root = tk.Tk()

btn = tk.Button(root, width=30, height=10, text="SPEAK", bg="red", command=record)
btn.pack(pady=20)

lbl = tk.Label(root, width=30, text="Guess:")
lbl.pack(pady=10)

timerlabel = tk.Label(root, width=30, text="5")
timerlabel.pack(pady=0)


btnlearn = tk.Button(root, width=30, height=5, text="learn", bg="yellow", command=guesser.learn)
btnlearn.pack(pady=40)
btnsave = tk.Button(root, width=30, height=5, text="save", bg="green", command=guesser.save)
btnsave.pack(pady=60)



root.mainloop()
