#!/usr/bin/env python3
"""
Détecteur de notes et accordeur chromatique en temps réel (DFT / FFT).

Chaîne de traitement :
  micro -> tampon glissant (N échantillons) -> fenêtre de Hann
        -> zero-padding -> DFT (FFT) -> spectre d'amplitude
        -> Harmonic Product Spectrum (HPS) -> pic -> interpolation parabolique
        -> fréquence fondamentale f0 -> note la plus proche + écart en cents

Usage :
  python tuner.py                      # utilise le micro
  python tuner.py --demo 440           # signal synthétique (sans micro)
  python tuner.py --wav melodie.wav    # analyse d'un fichier audio (sans micro)
  python tuner.py --demo 330 --save spectre.png   # sauvegarde une figure et quitte
Touche 's' dans la fenêtre : enregistre une capture d'écran PNG.
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

FS = 44100          # fréquence d'échantillonnage (Hz)
N = 8192            # taille de la fenêtre d'analyse (~186 ms)
ZERO_PAD = 2        # facteur de zero-padding (raffine la grille fréquentielle)
A4 = 440.0          # fréquence de référence du La4
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FMIN, FMAX = 60.0, 1200.0   # plage de recherche de f0 (Hz)
RMS_GATE = 0.01             # seuil de silence


# --------------------------------------------------------------------------- #
# DFT
# --------------------------------------------------------------------------- #
def dft_naive(x):
    """DFT directe, X[k] = sum_n x[n] e^{-2 i pi k n / N}.  Complexité O(N^2).
    Gardée pour illustrer la définition (utiliser N petit, ex. 512)."""
    n = len(x)
    idx = np.arange(n)
    W = np.exp(-2j * np.pi * np.outer(idx, idx) / n)
    return W @ x


def spectrum(frame, fs=FS):
    """Spectre d'amplitude (fenêtré + zero-padding) via la FFT, algorithme
    rapide (O(N log N)) qui calcule exactement la DFT."""
    n = len(frame)
    nfft = n * ZERO_PAD
    X = np.fft.rfft(frame * np.hanning(n), n=nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, np.abs(X)


# --------------------------------------------------------------------------- #
# Extraction de la fréquence fondamentale
# --------------------------------------------------------------------------- #
def estimate_f0(frame, fs=FS, fmin=FMIN, fmax=FMAX):
    """Retourne (f0, freqs, spec). f0 = None si le signal est trop faible."""
    freqs, spec = spectrum(frame, fs)
    if np.sqrt(np.mean(frame ** 2)) < RMS_GATE:
        return None, freqs, spec

    # Harmonic Product Spectrum : on multiplie le spectre par ses versions
    # décimées (x2, x3, x4). Les harmoniques s'alignent sur f0 -> pic net,
    # ce qui évite les erreurs d'octave quand l'harmonique 2 domine.
    L = len(spec) // 4
    hps = spec[:L].copy()
    for h in (2, 3, 4):
        hps *= spec[::h][:L]

    lo = np.searchsorted(freqs, fmin)
    hi = min(np.searchsorted(freqs, fmax), L - 2)
    k = lo + int(np.argmax(hps[lo:hi]))

    # Interpolation parabolique (sur le log du spectre) autour du pic
    a, b, c = np.log(spec[k - 1:k + 2] + 1e-12)
    denom = a - 2 * b + c
    p = 0.5 * (a - c) / denom if denom != 0 else 0.0
    f0 = (k + p) * fs / (len(frame) * ZERO_PAD)
    return float(f0), freqs, spec


def freq_to_note(f0, a4=A4):
    """Fréquence -> (nom de note, écart en cents, fréquence théorique)."""
    midi = 69 + 12 * np.log2(f0 / a4)
    n = int(round(midi))
    cents = 100 * (midi - n)
    name = f"{NOTES[n % 12]}{n // 12 - 1}"
    target = a4 * 2 ** ((n - 69) / 12)
    return name, cents, target


# --------------------------------------------------------------------------- #
# Source audio (micro ou signal de démonstration)
# --------------------------------------------------------------------------- #
def load_wav(path, fs=FS):
    """Charge un WAV PCM (8/16/32 bits), mélange en mono, rééchantillonne à fs."""
    import wave
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif sw == 2:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    elif sw == 4:
        x = np.frombuffer(raw, np.int32).astype(np.float32) / 2 ** 31
    else:
        raise SystemExit("Format WAV non supporté (utilisez PCM 16 bits).")
    x = x.reshape(-1, ch).mean(axis=1)
    if sr != fs:
        n_out = int(len(x) * fs / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)
    return x.astype(np.float32)


class Source:
    def __init__(self, demo_freq=None, fs=FS, wav=None):
        self.fs = fs
        self.buf = np.zeros(N, dtype=np.float32)
        self.demo = demo_freq
        self.t = 0
        self.stream = None
        self.wav = None
        if wav is not None:
            self.wav = load_wav(wav, fs)
        elif demo_freq is None:
            import sounddevice as sd
            self.stream = sd.InputStream(samplerate=fs, channels=1,
                                         blocksize=1024, callback=self._cb)
            self.stream.start()

    def _cb(self, indata, frames, t, status):
        self.push(indata[:, 0])

    def push(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.buf = np.concatenate((self.buf[len(x):], x))

    def read(self, n_new):
        if self.wav is not None:
            idx = (self.t + np.arange(n_new)) % len(self.wav)
            self.t += n_new
            self.push(self.wav[idx])
        elif self.demo is not None:
            t = (self.t + np.arange(n_new)) / self.fs
            self.t += n_new
            f = self.demo
            x = (0.6 * np.sin(2 * np.pi * f * t)
                 + 0.3 * np.sin(2 * np.pi * 2 * f * t)
                 + 0.15 * np.sin(2 * np.pi * 3 * f * t)
                 + 0.01 * np.random.randn(n_new))
            self.push(x)
        return self.buf.copy()


# --------------------------------------------------------------------------- #
# Interface graphique
# --------------------------------------------------------------------------- #
def build_figure():
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.35)
    ax_s = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])

    ax_s.set_xlim(0, 2000)
    ax_s.set_ylim(-90, 5)
    ax_s.set_xlabel("Fréquence (Hz)")
    ax_s.set_ylabel("Amplitude (dB)")
    ax_s.set_title("Spectre fréquentiel (DFT)")
    ax_s.grid(alpha=0.3)
    (line,) = ax_s.plot([], [], lw=1)
    vline = ax_s.axvline(0, color="r", ls="--", lw=1, visible=False)
    text = ax_s.text(0.98, 0.95, "--", transform=ax_s.transAxes, ha="right",
                     va="top", fontsize=22, weight="bold")

    ax_c.set_xlim(-50, 50)
    ax_c.set_ylim(0, 1)
    ax_c.set_yticks([])
    ax_c.set_xlabel("Écart (cents)  ← trop bas | trop haut →")
    ax_c.axvspan(-5, 5, color="green", alpha=0.25)
    ax_c.axvline(0, color="k", lw=1)
    needle = ax_c.axvline(0, color="r", lw=3, visible=False)
    return fig, line, vline, text, needle


def update_view(f0, freqs, spec, line, vline, text, needle, hist):
    db = 20 * np.log10(spec / (spec.max() + 1e-12) + 1e-5)
    line.set_data(freqs, db)
    if f0 is None:
        hist.clear()
        vline.set_visible(False)
        needle.set_visible(False)
        text.set_text("--")
        return
    hist.append(f0)
    del hist[:-5]
    f = float(np.median(hist))          # lissage par médiane
    name, cents, _ = freq_to_note(f)
    vline.set_xdata([f, f])
    vline.set_visible(True)
    needle.set_xdata([cents, cents])
    needle.set_visible(True)
    text.set_text(f"{name}   {f:.1f} Hz   {cents:+.0f} ¢")
    text.set_color("green" if abs(cents) < 5 else "black")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=float, help="fréquence d'un signal synthétique")
    ap.add_argument("--wav", help="fichier audio WAV à analyser (remplace le micro)")
    ap.add_argument("--save", help="sauvegarde une figure PNG puis quitte (avec --demo ou --wav)")
    args = ap.parse_args()

    src = Source(args.demo, wav=args.wav)
    fig, line, vline, text, needle = build_figure()
    hist = []

    if args.save:
        if args.demo is None and args.wav is None:
            raise SystemExit("--save nécessite --demo ou --wav")
        for _ in range(6):
            frame = src.read(N // 2)
        f0, freqs, spec = estimate_f0(frame)
        update_view(f0, freqs, spec, line, vline, text, needle, hist)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"f0 estimée = {f0:.2f} Hz -> {freq_to_note(f0)[0]} ; figure : {args.save}")
        return

    interval_ms = 60
    n_new = int(FS * interval_ms / 1000)

    def update(_):
        frame = src.read(n_new if src.stream is None else 0)
        f0, freqs, spec = estimate_f0(frame)
        update_view(f0, freqs, spec, line, vline, text, needle, hist)
        return line, vline, text, needle

    def on_key(event):
        if event.key == "s":
            name = time.strftime("capture_%Y%m%d_%H%M%S.png")
            fig.savefig(name, dpi=150, bbox_inches="tight")
            print("capture enregistrée :", name)

    fig.canvas.mpl_connect("key_press_event", on_key)
    _anim = FuncAnimation(fig, update, interval=interval_ms, blit=False,
                          cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()