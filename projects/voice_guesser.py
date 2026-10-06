import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import tkinter as tk
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import Cerebrum_v0


Voice_Ai = Cerebrum_v0.AI(2041, [64, 64, 32, 32, 32, 32, 3])


import tkinter as tk


root = tk.Tk()

btn = tk.Button(
    root,
    width=30,
    height=10,
    text="SPEAK",
    bg = "red"
)
btn.pack(pady=20)

lbl = tk.Label(
    root,
    width=30,
    text="Ready"
)
lbl.pack(pady=10)

root.mainloop()
