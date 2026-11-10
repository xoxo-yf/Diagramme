INSTALLATION:
pip install -r requirements.txt


UTILISATION:
python tuner.py                 # micro
python tuner.py --demo 440      # signal synthétique
python tuner.py --wav melodie.wav   # fichier audio (sans micro)
python tuner.py --demo 330 --save spectre.png



Principe
Tampon glissant de 8192 échantillons (44,1 kHz)
Fenêtre de Hann + zero-padding x2
DFT (FFT) -> spectre d'amplitude
Harmonic Product Spectrum -> pic = f0 ; interpolation parabolique
n = 69 + 12 log2(f0/440) -> note la plus proche, écart en cents