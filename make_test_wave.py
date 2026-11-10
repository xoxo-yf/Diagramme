"""Génère melodie.wav : les 6 cordes d'une guitare (E2 A2 D3 G3 B3 E4),
"""
import wave
import numpy as np

FS = 44100
notes = [82.41, 110.00, 146.83 * 2 ** (-20 / 1200), 196.00, 246.94, 329.63]
sig = []
for f in notes:
    t = np.arange(int(1.5 * FS)) / FS
    x = sum(a * np.sin(2 * np.pi * f * k * t) for k, a in enumerate([0.4, 0.6, 0.3, 0.15], 1))
    x *= np.exp(-1.2 * t)            # décroissance type corde pincée
    sig.append(x + 0.002 * np.random.randn(len(t)))
sig = np.concatenate(sig)
sig = (sig / np.abs(sig).max() * 0.8 * 32767).astype(np.int16)
with wave.open("melodie.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
    w.writeframes(sig.tobytes())
print("melodie.wav créé")