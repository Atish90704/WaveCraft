# WaveCraft — Audio DSP Workbench

A desktop audio editor and signal-processing toolkit built with Python (Tkinter + NumPy/SciPy), created for CSE 220 (Signals & Linear Systems). Combines classic DSP operations with a few applied/real-world extensions, all backed by manually implemented FFT, convolution, and correlation — not black-box audio libraries.

## Features
- **Edit & Trim** — load/record audio, trim, reverse, gain scale, fade in/out, undo
- **Convolution Effects** — smoothing (box-kernel low-pass) and echo, both true convolution
- **Space Simulator** — convolve with real recorded room impulse responses (OpenAIR) to simulate reverb of real spaces
- **FFT & Analysis** — spectrum/spectrogram view, pitch detection (note + cents), pitch-over-time piano roll, low/high-pass filters
- **AI Stem Separator** — isolate frequency bands via FFT masking, or full instrument stems via Demucs (neural network)
- **Parametric EQ** — 6-band frequency-domain equalizer with smooth log-frequency gain interpolation
- **Spectral Subtraction** — learns a noise sample's spectral fingerprint and subtracts it from the whole clip
- **Format Converter** — WAV/MP3/FLAC/OGG/M4A export
- **Sampling & Aliasing** — deliberately downsample below Nyquist to hear/see aliasing artifacts, vs. a properly anti-aliased downsample
- **Voice & Speaker ID** — register/identify speakers via MFCC voiceprints, persisted to disk across sessions
- **Doppler Speed Shift** — measure a passing sound's approach/departure frequencies to calculate source speed
- **BPM Pulse Detector** — spectral-flux onset detection + autocorrelation for song tempo or heart-rate BPM
- **Voice Stress Test** — jitter/shimmer pitch-stability analysis
- **Pulse-Echo Ranging** — audio "sonar": cross-correlation of an emitted pulse and its echo to measure distance
- Live click-to-seek waveform playback, light/dark theme, tooltips on every control

## How to Run
1. Install Python 3.10+
2. Install dependencies:
   ```
   pip install numpy scipy soundfile sounddevice pydub librosa requests
   ```
   (also needs `ffmpeg` on your system PATH for MP3/format conversion)
3. Run:
   ```
   python app.py
   ```

No build step — it's a single-window desktop app, sidebar-navigated by module.

Done by me - Atish(2305028) and Wasif(2305029) as our project in BUET CSE course 220, 2-2.
