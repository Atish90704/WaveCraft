"""
audio_ops.py
Core DSP functions for the audio editor.

All audio is handled as a 1D numpy float64 array in range [-1, 1]
(mono; stereo files get averaged down to mono on load). Every function
here is a pure function: (array in) -> (array out). This keeps the GUI
layer completely separate from the signal-processing layer, so you can
also just `import audio_ops` in a notebook and test things directly.
"""

import os
import tempfile
import subprocess
import json
import soundfile as sf
import numpy as np
import sounddevice as sd
import pitch_detector as pd
import yt_dlp
import librosa
import requests


from scipy.io import wavfile
from scipy.signal import fftconvolve, spectrogram as scipy_spectrogram, stft, istft, resample as scipy_signal_resample, correlate as scipy_correlate, chirp as scipy_chirp
from pydub import AudioSegment


# ---------------------------------------------------------------- I/O ----

def load_wav(path):
    """Read a WAV file. Returns (sample_rate, data) with data as float64
    in [-1, 1], mono (stereo channels are averaged)."""
    sr, data = wavfile.read(path)

    # wavfile gives ints (e.g. int16) or floats depending on the file.
    # Normalize whatever we get into float64 in [-1, 1].
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float64) / max_val
    else:
        data = data.astype(np.float64)

    if data.ndim == 2:  # stereo -> mono
        data = data.mean(axis=1)

    return sr, data


def save_wav(path, sr, data):
    """Write a float64 [-1, 1] array back out as 16-bit PCM WAV."""
    data = np.clip(data, -1.0, 1.0)
    int_data = (data * 32767).astype(np.int16)
    wavfile.write(path, sr, int_data)


def save_audio(path, sr, data, fmt=None):
    """Save as WAV natively, or any other format (MP3, FLAC, OGG, M4A/AAC)
    via pydub + ffmpeg. `fmt` is the export format string, e.g. 'mp3';
    if omitted it's inferred from the file extension."""
    if fmt is None:
        fmt = path.rsplit(".", 1)[-1].lower()

    if fmt == "wav":
        save_wav(path, sr, data)
        return

    # ffmpeg doesn't recognize "m4a" as a muxer name directly - it needs
    # "ipod" (the actual container format name for .m4a files).
    export_fmt = "ipod" if fmt == "m4a" else fmt

    data = np.clip(data, -1.0, 1.0)
    int_data = (data * 32767).astype(np.int16)
    audio = AudioSegment(
        int_data.tobytes(),
        frame_rate=sr,
        sample_width=2,   # 16-bit
        channels=1,       # mono, matches how we load everything
    )
    audio.export(path, format=export_fmt)

#--------------------------------------------------------mp3--------------
def load_mp3(path):
    audio = AudioSegment.from_mp3(path)
    sr = audio.frame_rate
    data = np.array(audio.get_array_of_samples()).astype(np.float64)
    if audio.channels == 2:
        data = data.reshape((-1, 2)).mean(axis=1)
    data /= (2**(8*audio.sample_width - 1))  # normalize to [-1,1]
    return sr, data


def load_audio(path):
    """Loads WAV natively, or any other format (MP3, FLAC, OGG, M4A) via pydub."""
    if path.lower().endswith(".wav"):
        return load_wav(path)
    audio = AudioSegment.from_file(path)  # Handles MP3, FLAC, OGG, M4A, AAC automatically
    sr = audio.frame_rate
    data = np.array(audio.get_array_of_samples()).astype(np.float64)
    if audio.channels == 2:
        data = data.reshape((-1, 2)).mean(axis=1)
    data /= (2**(8 * audio.sample_width - 1))
    return sr, data

#===============================audio recorder=======================
class AudioRecorder:
    """Handles background microphone recording using sounddevice streams."""
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.frames = []
        self.recording = False
        self.stream = None

    def start(self):
        self.frames = []
        self.recording = True
        def callback(indata, frames, time, status):
            if self.recording:
                self.frames.append(indata.copy())

        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback)
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.frames:
            # Stack audio buffers into a flat float64 numpy array
            data = np.concatenate(self.frames, axis=0).squeeze().astype(np.float64)
            return self.sample_rate, data
        return self.sample_rate, np.array([], dtype=np.float64)

# ------------------------------------------------------------ editing ----

def trim(data, sr, start_sec, end_sec):
    """Keep only [start_sec, end_sec]."""
    start = max(0, int(start_sec * sr))
    end = min(len(data), int(end_sec * sr))
    if start >= end:
        raise ValueError("start must be before end")
    return data[start:end]


def join(data_a, data_b):
    """Concatenate two clips back to back. Caller must ensure same sample rate."""
    return np.concatenate([data_a, data_b])


def reverse(data):
    return data[::-1].copy()


def scale(data, factor):
    """Change volume/amplitude by `factor` (e.g. 0.5 = quieter, 2.0 = louder).
    Clipped to [-1, 1] on save, but we clip here too so waveform preview matches."""
    return np.clip(data * factor, -1.0, 1.0)


def fade(data, sr, fade_in_sec=0.0, fade_out_sec=0.0):
    """Linear fade in at the start and/or fade out at the end."""
    out = data.copy()
    n = len(out)

    n_in = int(fade_in_sec * sr)
    if n_in > 0:
        n_in = min(n_in, n)
        ramp = np.linspace(0.0, 1.0, n_in)
        out[:n_in] *= ramp

    n_out = int(fade_out_sec * sr)
    if n_out > 0:
        n_out = min(n_out, n)
        ramp = np.linspace(1.0, 0.0, n_out)
        out[n - n_out:] *= ramp

    return out


# ------------------------------------------------- convolution effects ----
#
# Both effects below work the same way: build a short "impulse response"
# h[n], then run y = x * h (linear convolution). This is exactly the
# LTI system theory from the course - an audio effect IS a system, and
# applying it IS convolution.

def smooth(data, kernel_size=9):
    """Moving-average smoothing (a low-pass filter) via convolution.
    Larger kernel_size = more smoothing = duller/muffled sound.
    h[n] is a box of height 1/kernel_size, width kernel_size."""
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")
    h = np.ones(kernel_size) / kernel_size
    return fftconvolve(data, h, mode="same")


def convolve_with_ir(data, sr, ir, ir_sr, wet=1.0):
    """Convolve `data` with a real recorded impulse response `ir` (a 1D
    mono float array, same idea as smooth()/echo() above but with a
    measured room/hall/plate recording standing in for the synthetic
    boxcar or spike-train impulse response).

    ir_sr: the impulse response's own sample rate. If it doesn't match
    the track's `sr`, the IR is resampled first so the reverb tail's
    timing is correct instead of being sped up/slowed down.

    wet: 0..1 dry/wet mix - 1.0 is fully "wet" (just the convolution
    result), lower values blend back in some of the original dry signal
    so the effect isn't overwhelming.
    """
    if ir is None or len(ir) == 0:
        raise ValueError("Impulse response is empty.")
    ir = np.asarray(ir, dtype=np.float64)
    if ir_sr != sr:
        n_target = int(round(len(ir) * sr / ir_sr))
        ir = scipy_signal_resample(ir, max(n_target, 1))
    ir = ir / (np.max(np.abs(ir)) or 1.0)  # normalize IR energy before convolving

    wet_signal = fftconvolve(data, ir, mode="full")[:len(data) + len(ir) - 1]

    # normalize so the convolution doesn't blow past [-1, 1]
    peak = np.max(np.abs(wet_signal)) if len(wet_signal) else 1.0
    if peak > 1.0:
        wet_signal = wet_signal / peak

    dry_padded = np.zeros_like(wet_signal)
    dry_padded[:len(data)] = data
    wet = max(0.0, min(1.0, wet))
    return wet * wet_signal + (1.0 - wet) * dry_padded


def fetch_impulse_response(url: str, progress_hook=None):
    """Downloads an impulse-response audio file from a direct URL (e.g. a
    file link copied from an OpenAIR (openair.hosted.york.ac.uk) recording
    page) and returns (sample_rate, data) ready to hand to
    convolve_with_ir().

    OpenAIR's recordings are published specifically for reuse in exactly
    this kind of application (auralization / convolution reverb), so
    there's no ToS concern the way there is with stream-ripping YouTube -
    same reasoning as fetch_from_archive_org() above.

    There's no documented search-by-name API for OpenAIR, so this takes
    a direct file URL (copy it from the "Download" link on the
    recording's page on openair.hosted.york.ac.uk) rather than a song
    name - unlike fetch_from_archive_org(), which can search by name.
    """
    if not url or not url.strip():
        raise ValueError("Enter a direct impulse-response file URL first.")

    def _notify(status, **extra):
        if progress_hook is not None:
            progress_hook({"status": status, **extra})

    _notify("downloading", url=url)
    file_name = os.path.basename(url.split("?")[0]) or "impulse_response.wav"

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, file_name)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_path, "wb") as out_f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    out_f.write(chunk)

        _notify("converting")
        sr, data = load_audio(local_path)
        return sr, data, file_name


def echo(data, sr, delay_sec=0.3, decay=0.5, n_repeats=4):
    """Echo effect via convolution with an impulse response made of
    decaying spikes spaced delay_sec apart:
        h = [1, 0, ..., 0, decay, 0, ..., 0, decay^2, ...]
                 ^ delay_sec        ^ 2*delay_sec
    Convolving the signal with this h automatically produces the
    original signal plus decaying delayed copies of itself - an echo."""
    delay_samples = int(delay_sec * sr)
    if delay_samples < 1:
        raise ValueError("delay_sec too small for this sample rate")

    h_len = delay_samples * n_repeats + 1
    h = np.zeros(h_len)
    h[0] = 1.0
    for k in range(1, n_repeats + 1):
        h[k * delay_samples] = decay ** k

    y = fftconvolve(data, h, mode="full")
    y = y[:len(data) + delay_samples * n_repeats]  # keep the tail (the echoes)

    # normalize so the echo doesn't blow past [-1, 1]
    peak = np.max(np.abs(y)) if len(y) else 1.0
    if peak > 1.0:
        y = y / peak

    return y


# ------------------------------------------------ frequency-domain (FFT) ----
#
# Everything below works in the frequency domain instead of the time domain.
# rfft turns the signal into its frequency components (DFT/FFT theory from
# the course); irfft turns it back. Filtering here means zeroing out the
# frequency components you don't want, which is a "proper" version of the
# time-domain moving-average smoothing above.

def compute_spectrum(data, sr):
    """FFT magnitude spectrum. Returns (freqs, magnitude) - one value per
    frequency bin from 0 Hz up to sr/2 (Nyquist)."""
    n = len(data)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    spectrum = np.fft.rfft(data)
    magnitude = np.abs(spectrum) / n
    return freqs, magnitude


def compute_spectrogram(data, sr, nperseg=1024):
    """How the frequency content changes over time.
    Returns (freqs, times, Sxx) where Sxx[i, j] is the power at freqs[i], times[j]."""
    f, t, Sxx = scipy_spectrogram(data, fs=sr, nperseg=nperseg)
    return f, t, Sxx


def lowpass_filter(data, sr, cutoff_hz):
    """Keep only frequencies below cutoff_hz (removes hiss/treble)."""
    n = len(data)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    spectrum = np.fft.rfft(data)
    spectrum[freqs > cutoff_hz] = 0
    return np.fft.irfft(spectrum, n=n)


def highpass_filter(data, sr, cutoff_hz):
    """Keep only frequencies above cutoff_hz (removes rumble/bass)."""
    n = len(data)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    spectrum = np.fft.rfft(data)
    spectrum[freqs < cutoff_hz] = 0
    return np.fft.irfft(spectrum, n=n)


def bandstop_filter(data, sr, low_hz, high_hz):
    """Remove frequencies in [low_hz, high_hz], keep everything else.
    This is the 'decompose' operation: e.g. to roughly remove a vocal
    or instrument, zero out the frequency range it mostly lives in."""
    n = len(data)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    spectrum = np.fft.rfft(data)
    spectrum[(freqs >= low_hz) & (freqs <= high_hz)] = 0
    return np.fft.irfft(spectrum, n=n)


def bandpass_filter(data, sr, low_hz, high_hz):
    """Keep only frequencies in [low_hz, high_hz], zero everything else.
    Useful to isolate roughly where a voice/instrument lives, e.g. to
    solo it for inspection."""
    n = len(data)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    spectrum = np.fft.rfft(data)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0
    return np.fft.irfft(spectrum, n=n)


    #=============extra function for frequency detection===============
def detect_pitch(data, sr):
    """Calculates FFT, finds dominant peak frequency, and maps to a musical note."""
    if len(data) == 0:
        return {"is_silent": True}
    
    freqs, mag = compute_spectrum(data, sr)
    if len(mag) <= 1:
        return {"is_silent": True}
    
    # Ignore DC component (0 Hz bin) and find peak magnitude index
    max_idx = np.argmax(mag[1:]) + 1
    max_val = mag[max_idx]
    
    # Silence threshold
    if max_val < 0.0001:
        return {"is_silent": True}
        
    dominant_freq = freqs[max_idx]
    note_info = pd.frequency_to_note(dominant_freq)
    
    return {
        "frequency": round(dominant_freq, 1),
        "note_info": note_info,
        "is_silent": False
    }


def detect_doppler_shift(data, sr):
    """Automate the approach/depart measurement: split the clip at its
    loudest sample (the closest pass-by moment - a source is loudest right
    as it passes you), treat everything before that point as 'approaching'
    and everything after as 'departing', and run detect_pitch on each half.

    Returns (f_approach, f_depart). Raises ValueError if the clip is too
    short, the peak sits too close to either edge (not enough audio on
    one side to analyze), or either half is too quiet to detect a pitch.
    """
    if len(data) < 4:
        raise ValueError("Clip too short to analyze.")

    peak_idx = int(np.argmax(np.abs(data)))
    approach_seg = data[:peak_idx]
    depart_seg = data[peak_idx:]

    min_len = int(0.05 * sr)  # need at least ~50ms of audio on each side
    if len(approach_seg) < min_len or len(depart_seg) < min_len:
        raise ValueError(
            "The loudest point is too close to the start/end of the clip - "
            "trim the recording so there's audio before AND after the pass-by moment."
        )

    approach_info = detect_pitch(approach_seg, sr)
    depart_info = detect_pitch(depart_seg, sr)

    if approach_info.get("is_silent"):
        raise ValueError("The 'approaching' half of the clip is too quiet to detect a pitch.")
    if depart_info.get("is_silent"):
        raise ValueError("The 'departing' half of the clip is too quiet to detect a pitch.")

    return approach_info["frequency"], depart_info["frequency"]


# ------------------------------------------------------ voice stress test ----
#
# Frame-by-frame pitch (jitter) and amplitude (shimmer) stability, the same
# core idea as clinical voice-tension analysis - pure DSP, no ML model.
# Pitch is estimated per-frame with autocorrelation rather than the FFT-peak
# method above, because autocorrelation is far more reliable on the very
# short (20-50ms) windows this needs.

def _autocorr_pitch_frame(frame, sr, fmin=75.0, fmax=500.0):
    """Estimates the fundamental frequency of one short frame via
    autocorrelation. Returns None if the frame looks unvoiced/silent
    (no clear periodicity)."""
    frame = frame - np.mean(frame)
    energy = np.sum(frame ** 2)
    if energy < 1e-6:
        return None

    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2:]  # keep only non-negative lags
    corr = corr / (corr[0] + 1e-12)

    min_lag = int(sr / fmax)
    max_lag = min(int(sr / fmin), len(corr) - 1)
    if max_lag <= min_lag:
        return None

    search = corr[min_lag:max_lag]
    peak_idx = int(np.argmax(search))
    peak_val = search[peak_idx]
    if peak_val < 0.3:  # too weak/aperiodic -> treat as unvoiced
        return None

    lag = min_lag + peak_idx
    return sr / lag


def analyze_voice_stress(data, sr, frame_ms=30.0, hop_ms=15.0):
    """
    Tracks pitch (f0) and RMS amplitude frame-by-frame across the clip,
    then computes:
      - jitter:  average frame-to-frame relative change in pitch PERIOD
                 across consecutive VOICED frames (classic jitter definition)
      - shimmer: average frame-to-frame relative change in amplitude
                 across the same frames

    Higher jitter/shimmer than a calm baseline suggests vocal tension -
    voices naturally waver a little; a shaky/stressed voice wavers more.

    Returns a dict: {times, f0_track (nan for unvoiced frames), amp_track,
    jitter_percent, shimmer_percent, stress_score, stress_label}.
    """
    if data is None or len(data) == 0:
        raise ValueError("No audio loaded.")

    frame_len = max(int(sr * frame_ms / 1000.0), 32)
    hop_len = max(int(sr * hop_ms / 1000.0), 1)

    times, f0_track, amp_track = [], [], []
    for start in range(0, len(data) - frame_len, hop_len):
        frame = data[start:start + frame_len]
        f0 = _autocorr_pitch_frame(frame, sr)
        rms = float(np.sqrt(np.mean(frame ** 2)))
        times.append(start / sr)
        f0_track.append(f0 if f0 is not None else np.nan)
        amp_track.append(rms)

    times = np.array(times)
    f0_track = np.array(f0_track)
    amp_track = np.array(amp_track)

    voiced_mask = ~np.isnan(f0_track)
    n_voiced = int(np.sum(voiced_mask))
    if n_voiced < 3:
        raise ValueError("Not enough sustained voiced sound detected to analyze "
                          "(try a longer clip of continuous speech/vowel sound).")

    periods = 1.0 / f0_track[voiced_mask]
    amps = amp_track[voiced_mask]

    period_diffs = np.abs(np.diff(periods))
    jitter_percent = 100.0 * np.mean(period_diffs) / np.mean(periods)

    amp_diffs = np.abs(np.diff(amps))
    mean_amp = np.mean(amps)
    shimmer_percent = 100.0 * np.mean(amp_diffs) / mean_amp if mean_amp > 1e-9 else 0.0

    # Rough, non-clinical thresholds for a friendly demo readout:
    # typical calm speech sits well under these; stress/tension pushes both up.
    stress_score = min(100.0, (jitter_percent / 2.0 + shimmer_percent / 8.0) * 25.0)
    if stress_score < 35:
        stress_label = "Low (calm/steady voice)"
    elif stress_score < 65:
        stress_label = "Moderate"
    else:
        stress_label = "High (unstable pitch/volume)"

    return {
        "times": times,
        "f0_track": f0_track,
        "amp_track": amp_track,
        "jitter_percent": round(jitter_percent, 3),
        "shimmer_percent": round(shimmer_percent, 3),
        "stress_score": round(stress_score, 1),
        "stress_label": stress_label,
    }


# ------------------------------------------------------ pulse-echo ranging ----
#
# The audio version of sonar/radar: emit a known chirp, cross-correlate the
# recording against it to find each copy's time delay, and convert delay to
# distance via the speed of sound. A cross-correlation peak marks where the
# emitted pulse's "shape" lines up best with the recording - the strongest
# peak is normally the direct speaker-to-mic path, and a later, weaker peak
# is a reflection.

SPEED_OF_SOUND_M_S = 343.0  # in air at ~20C


def generate_chirp(sr=44100, duration_sec=0.02, f0=2000.0, f1=9000.0):
    """Generates a short linear frequency sweep ('chirp') - a good pulse
    for ranging because its distinct, non-repeating shape correlates far
    more sharply/unambiguously than a plain click or tone would."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    pulse = scipy_chirp(t, f0=f0, f1=f1, t1=duration_sec, method="linear")
    # short fade in/out so the pulse doesn't click the speaker
    fade_len = max(int(0.1 * len(pulse)), 1)
    window = np.ones(len(pulse))
    window[:fade_len] = np.linspace(0, 1, fade_len)
    window[-fade_len:] = np.linspace(1, 0, fade_len)
    return (pulse * window).astype(np.float64)


def _find_correlation_peaks(corr, min_separation, threshold_ratio=0.2):
    """Finds the global max, then the next local max at least
    `min_separation` samples later that's still a meaningfully strong
    echo (>= threshold_ratio of the global max)."""
    global_idx = int(np.argmax(corr))
    global_val = corr[global_idx]

    search_start = global_idx + min_separation
    if search_start >= len(corr) - 1:
        return global_idx, None

    tail = corr[search_start:]
    echo_idx_rel = int(np.argmax(tail))
    echo_val = tail[echo_idx_rel]
    if echo_val < threshold_ratio * global_val:
        return global_idx, None
    return global_idx, search_start + echo_idx_rel


def measure_echo_delay(emitted, received, sr, min_gap_ms=3.0):
    """
    Cross-correlates `received` (mic recording, containing both the direct
    speaker sound and any echo) against `emitted` (the known pulse). The
    strongest correlation peak is the direct path; the next strong peak at
    least min_gap_ms later is treated as the echo.

    Returns dict: {direct_lag_sec, echo_lag_sec, delay_sec, distance_m,
    correlation, lags_sec} - the last two are for plotting.
    Raises ValueError if no distinct echo peak is found.
    """
    if emitted is None or received is None or len(emitted) == 0 or len(received) == 0:
        raise ValueError("Need both an emitted pulse and a recorded signal.")

    corr = scipy_correlate(received, emitted, mode="full")
    lags = np.arange(-len(emitted) + 1, len(received))
    lags_sec = lags / sr

    min_gap_samples = max(int(sr * min_gap_ms / 1000.0), 1)
    direct_idx, echo_idx = _find_correlation_peaks(corr, min_gap_samples)

    if echo_idx is None:
        raise ValueError(
            "No distinct second echo found - either there's no reflective surface in range, or it's "
            "too close: a short-range echo can overlap the direct pulse. Try a farther/harder wall, "
            "or shorten the pulse duration."
        )

    delay_sec = (echo_idx - direct_idx) / sr
    if delay_sec <= 0:
        raise ValueError("Echo appears to arrive before the direct sound - invalid recording.")

    distance_m = (delay_sec * SPEED_OF_SOUND_M_S) / 2.0

    return {
        "direct_lag_sec": lags_sec[direct_idx],
        "echo_lag_sec": lags_sec[echo_idx],
        "delay_sec": delay_sec,
        "distance_m": distance_m,
        "correlation": corr,
        "lags_sec": lags_sec,
    }


def simulate_echo(emitted, sr, distance_m, attenuation=0.4, noise_level=0.01):
    """
    Builds a SYNTHETIC 'received' signal for testing the ranging math
    without needing a real room/mic: the direct pulse at t=0, plus a
    quieter, delayed copy simulating a reflection from `distance_m` away,
    plus a little background noise. Feed the result into
    measure_echo_delay() to verify the recovered distance matches.
    """
    if distance_m <= 0:
        raise ValueError("distance_m must be positive.")
    delay_sec = (2.0 * distance_m) / SPEED_OF_SOUND_M_S
    delay_samples = int(round(delay_sec * sr))

    total_len = len(emitted) + delay_samples + len(emitted)
    received = np.zeros(total_len, dtype=np.float64)
    received[:len(emitted)] += emitted  # direct path
    received[delay_samples:delay_samples + len(emitted)] += attenuation * emitted  # echo

    if noise_level > 0:
        received += np.random.normal(0, noise_level, size=received.shape)

    return received


# ------------------------------------------------------------ equalizer ----
#
# A "real" EQ, as opposed to the hard on/off lowpass/highpass/bandpass
# filters above: instead of zeroing frequencies out, it scales each band
# up or down by a gain (in dB). Band edges are fixed; gains are supplied
# by the caller (e.g. from GUI sliders).

EQ_BANDS = [
    ("Sub Bass", 20, 60),
    ("Bass", 60, 250),
    ("Low Mid", 250, 500),
    ("Mid", 500, 2000),
    ("High Mid", 2000, 4000),
    ("Treble", 4000, 20000),
]


def equalize(data, sr, gains_db):
    """Multi-band equalizer. gains_db is a list of dB values, one per
    band in EQ_BANDS in order (0 = unchanged, +N boosts, -N cuts).

    Works in the frequency domain like the other filters, but instead
    of a hard cutoff, each FFT bin's gain is found by interpolating
    between band-center gains in log-frequency space. That gives a
    smooth EQ curve (like a real equalizer's shelving/peaking bands)
    instead of the boxy artifacts a hard per-band cutoff would cause."""
    if len(gains_db) != len(EQ_BANDS):
        raise ValueError(f"expected {len(EQ_BANDS)} gain values, got {len(gains_db)}")

    n = len(data)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    spectrum = np.fft.rfft(data)

    # Band "centers" (geometric mean, since we interpolate in log-freq).
    centers = np.array([np.sqrt(lo * hi) for _, lo, hi in EQ_BANDS])
    gains_linear = np.array([10 ** (g / 20.0) for g in gains_db])

    safe_freqs = np.where(freqs <= 0, 1e-6, freqs)  # avoid log(0) at DC
    gain_curve = np.interp(
        np.log10(safe_freqs), np.log10(centers), gains_linear,
        left=gains_linear[0], right=gains_linear[-1],
    )

    spectrum = spectrum * gain_curve
    return np.fft.irfft(spectrum, n=n)


# --------------------------------------------------------- noise removal ----
#
# Spectral subtraction: point the tool at a stretch of the clip that is
# JUST noise (room hiss, hum, fan noise - no wanted signal), it learns
# that noise's average spectral "fingerprint", then subtracts that
# fingerprint's magnitude from every frame of the full clip's STFT.
# Phase is left untouched (subtracting only affects magnitude).

def estimate_noise_profile(data, sr, start_sec, end_sec, nperseg=1024):
    """Average magnitude spectrum of a noise-only region [start_sec, end_sec].
    Pass the result into remove_noise()."""
    start = max(0, int(start_sec * sr))
    end = min(len(data), int(end_sec * sr))
    if start >= end:
        raise ValueError("start must be before end")
    if end - start < nperseg:
        raise ValueError("noise sample region is too short - pick a longer stretch")

    noise_clip = data[start:end]
    _, _, Zxx = stft(noise_clip, fs=sr, nperseg=nperseg)
    return np.mean(np.abs(Zxx), axis=1)  # avg magnitude per frequency bin


def remove_noise(data, sr, noise_profile, nperseg=1024, strength=1.5, floor=0.05):
    """Subtract `strength` * noise_profile from every STFT frame's
    magnitude, keeping phase unchanged. Each bin is floored at
    `floor` * its original magnitude rather than allowed to hit zero -
    over-subtracting all the way to silence creates 'musical noise'
    (random watery/robotic artifacts), so a small residual is kept
    on purpose. `strength` > 1 subtracts more aggressively (more noise
    removed, more risk of damaging the wanted signal)."""
    if len(noise_profile) != nperseg // 2 + 1:
        raise ValueError("noise_profile doesn't match nperseg - re-estimate it with the same nperseg")

    f, t, Zxx = stft(data, fs=sr, nperseg=nperseg)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    clean_mag = mag - strength * noise_profile[:, None]
    clean_mag = np.maximum(clean_mag, floor * mag)

    Zxx_clean = clean_mag * np.exp(1j * phase)
    _, cleaned = istft(Zxx_clean, fs=sr, nperseg=nperseg)
    return cleaned[:len(data)]


#===============piano playing feature================
def track_pitch_over_time(data, sr, window_sec=0.1, hop_sec=0.25, min_energy_ratio=0.03):
    """
    Samples audio every 0.25 seconds (4 frames per second).
    Filters out silence and restricts pitch output to human musical ranges.
    """
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    times = []
    midi_notes = []
    
    if len(data) < window_samples:
        return np.array([]), np.array([])

    overall_rms = np.sqrt(np.mean(data**2)) if len(data) > 0 else 1.0
    silence_threshold = overall_rms * min_energy_ratio

    for start in range(0, len(data) - window_samples, hop_samples):
        frame = data[start : start + window_samples]
        t = (start + window_samples / 2) / sr
        
        frame_rms = np.sqrt(np.mean(frame**2))
        if frame_rms < silence_threshold:
            times.append(t)
            midi_notes.append(np.nan)
            continue
            
        windowed = frame * np.hanning(len(frame))
        pitch = detect_pitch(windowed, sr)
        
        times.append(t)
        if pitch["is_silent"]:
            midi_notes.append(np.nan)
        else:
            freq = pitch["frequency"]
            if 65.0 <= freq <= 2100.0:  # C2 to C7 range
                midi = 12 * np.log2(freq / 440.0) + 69
                midi_notes.append(midi)
            else:
                midi_notes.append(np.nan)
                
    return np.array(times), np.array(midi_notes)



#================demucs=================
def separate_audio_demucs(data, sr, stem="vocals"):
    """
    Separates stems using Meta's Demucs model (PyTorch).
    Available stems: 'vocals', 'drums', 'bass', 'other'
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.wav")
        sf.write(input_path, data, sr)

        # Execute Demucs CLI via Python
        cmd = [
            "python", "-m", "demucs",
            "-n", "htdemucs",
            "--out", temp_dir,
            input_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Demucs failed: {result.stderr}")

        # Demucs places files inside: {temp_dir}/htdemucs/input/{stem}.wav
        output_path = os.path.join(temp_dir, "htdemucs", "input", f"{stem}.wav")
        
        if os.path.exists(output_path):
            stem_data, out_sr = sf.read(output_path)
            if stem_data.ndim > 1:
                stem_data = stem_data.mean(axis=1)  # Convert stereo to mono
            return stem_data, out_sr
        else:
            raise FileNotFoundError(f"Could not locate extracted stem: {stem}")

# Rough, commonly-cited frequency ranges (Hz) for the "decompose" presets.
# These are approximate - real instruments have overlapping harmonics, so
# removing one range will always bleed into others a little. That's an
# inherent limitation of frequency-domain source separation without ML,
# not a bug. Ranges are the widely-used "mixing engineer" fundamental
# ranges for each instrument.
DECOMPOSE_PRESETS = {
    "Vocals":            (85, 1100),
    "Bass Guitar":       (40, 400),
    "Kick Drum":         (50, 100),
    "Snare Drum":        (120, 500),
    "Hi-Hat / Cymbals":  (5000, 15000),
    "Guitar":            (80, 1200),
    "Piano":             (27, 4200),
    "Strings / Violin":  (196, 3500),
    "Trumpet / Brass":   (165, 1000),
    "Saxophone":         (100, 900),
}


# ---------------------------------------------------------------- BPM ----

def compute_bpm(data, sr, band=None, min_bpm=40, max_bpm=220):
    """Estimate tempo (beats per minute) via onset-envelope autocorrelation.

    band: optional (low_hz, high_hz) to band-limit before extracting the
    envelope - e.g. band=(20, 150) for a heartbeat-style signal instead
    of a full song.

    Note: this counts *any* repeating loud pulse in the chosen band, not
    "musical beats" specifically - and for heartbeats specifically, a
    real heartbeat has two sounds per cycle (S1 "lub" + S2 "dub"), which
    this simple onset counter can't tell apart from two separate beats.
    So heartbeat BPM here is a rough pulse-rate estimate, not a clinical
    reading.
    """
    x = data.copy()
    if band is not None:
        x = bandpass_filter(x, sr, band[0], band[1])

    # Onset envelope: rectify, then smooth with a ~50ms moving average
    envelope = np.abs(x)
    win = max(1, int(0.05 * sr))
    kernel = np.ones(win) / win
    envelope = fftconvolve(envelope, kernel, mode="same")

    # Downsample the envelope - we only need ~100-200Hz resolution to find
    # a beat, and it makes the autocorrelation much cheaper on long clips.
    down_factor = max(1, sr // 200)
    env_ds = envelope[::down_factor]
    sr_ds = sr / down_factor
    env_ds = env_ds - np.mean(env_ds)

    if len(env_ds) < 2:
        return None

    corr = np.correlate(env_ds, env_ds, mode="full")
    corr = corr[len(corr) // 2:]

    min_lag = int(sr_ds * 60 / max_bpm)
    max_lag = int(sr_ds * 60 / min_bpm)
    max_lag = min(max_lag, len(corr) - 1)
    if min_lag >= max_lag:
        return None

    search = corr[min_lag:max_lag]
    if len(search) == 0 or np.max(search) <= 0:
        return None

    peak_lag = np.argmax(search) + min_lag
    bpm = 60 * sr_ds / peak_lag
    return round(bpm, 1)


# --------------------------------------------------------- voice detect ----

# ------------------------------------------------- speaker identification ----
#
# "Who is talking" (not just "is someone talking") via classic MFCC
# voiceprints: Mel-Frequency Cepstral Coefficients summarize a voice's
# timbre/formant shape into a small vector. Enroll a few known speakers
# (extract + store their MFCC vector), then compare a new clip's MFCC
# vector against each enrolled one - closest match (by distance) wins.
# This is the standard classical (pre-deep-learning) approach to speaker
# ID; it's not as accurate as a trained neural embedding model, but it's
# real signal-processing, not a black box.

def extract_mfcc(data, sr, n_mfcc=13):
    """Extract a single MFCC 'voiceprint' vector for a clip: the standard
    mel-cepstral pipeline (mel filterbank -> log -> DCT), averaged across
    time so the whole clip collapses to one n_mfcc-length vector."""
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "Speaker ID needs the 'librosa' package - install with: pip install librosa"
        )
    y = data.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)


def identify_speaker(data, sr, enrolled_voiceprints, n_mfcc=13, threshold=40.0):
    """Compare this clip's MFCC voiceprint against every enrolled speaker
    and return the closest match.

    enrolled_voiceprints: dict of {name: mfcc_vector} (from extract_mfcc).
    threshold: max Euclidean distance still counted as a confident match;
    beyond that, returns "Unknown" rather than forcing a guess.

    Returns (name_or_"Unknown"_or_None, distance). None only if no
    speakers are enrolled yet.
    """
    if not enrolled_voiceprints:
        return None, None

    vec = extract_mfcc(data, sr, n_mfcc=n_mfcc)

    best_name = None
    best_dist = float("inf")
    for name, ref_vec in enrolled_voiceprints.items():
        dist = float(np.linalg.norm(vec - ref_vec))
        if dist < best_dist:
            best_dist = dist
            best_name = name

    if best_dist > threshold:
        return "Unknown", best_dist
    return best_name, best_dist


def detect_voice_activity(data, sr, window_sec=0.05, hop_sec=0.025,
                           voice_band=(300, 3400), energy_ratio_threshold=0.35):
    """Rough voice-activity detector. For each short window, computes what
    fraction of the window's energy falls inside the typical speech band
    (300-3400 Hz, the classic telephone-bandwidth range). Windows above
    the ratio threshold are flagged as containing voice.

    This is an energy-in-band heuristic, not true speech recognition - a
    sustained flute note or a synth pad sitting in the same band can also
    trigger it. It's the same category of limitation as the Decompose
    feature: frequency-domain, not source-aware.

    Returns (times, is_voice_bool_array, voice_percentage).
    """
    n = len(data)
    win = max(1, int(window_sec * sr))
    hop = max(1, int(hop_sec * sr))
    hann = np.hanning(win)

    times = []
    flags = []
    for start in range(0, max(1, n - win), hop):
        frame = data[start:start + win] * hann
        spec = np.fft.rfft(frame)
        freqs = np.fft.rfftfreq(win, d=1 / sr)
        mag2 = np.abs(spec) ** 2
        total = mag2.sum() + 1e-12
        band_energy = mag2[(freqs >= voice_band[0]) & (freqs <= voice_band[1])].sum()
        ratio = band_energy / total
        times.append(start / sr)
        flags.append(ratio >= energy_ratio_threshold)

    times = np.array(times)
    flags = np.array(flags, dtype=bool)
    pct = 100.0 * flags.mean() if len(flags) else 0.0
    return times, flags, pct


# ------------------------------------------------------------- doppler ----

def doppler_speed(f_approach, f_depart, speed_of_sound=343.0):
    """Doppler-shift source speed, solved from two OBSERVED frequencies -
    one measured while the source was approaching, one while it was
    departing - with no need to know/assume the horn's true frequency.

    Derivation: f_approach = f0*v/(v - vs), f_depart = f0*v/(v + vs).
    Dividing the two equations cancels out f0 entirely, leaving:
        vs = v * (f_approach - f_depart) / (f_approach + f_depart)

    This is more honest than assuming a source frequency from a spec
    sheet - it uses only what you actually measured. Returns speed in
    m/s (always positive magnitude, since "approach" vs "depart" already
    encodes the direction). speed_of_sound defaults to 343 m/s (dry air,
    ~20C).
    """
    if f_approach <= 0 or f_depart <= 0:
        raise ValueError("Frequencies must be positive")
    if f_approach <= f_depart:
        raise ValueError("Approaching frequency must be higher than departing frequency")

    v_s = speed_of_sound * (f_approach - f_depart) / (f_approach + f_depart)
    return v_s


#==============fetch youtube=========

def fetch_youtube_by_song_name(song_name: str, progress_hook=None):
    """
    Searches YouTube for `song_name` (ytsearch1 = top result only), downloads
    that video's audio, converts it to WAV, and returns (sample_rate, data)
    - the same (sr, data) order as load_audio()/load_wav(), so the result
    can be dropped straight into self.sr / self.data in the GUI.

    Downloads into a temp directory that is cleaned up automatically, so no
    leftover "temp_yt_audio.*" files are left sitting in the project folder.

    progress_hook: optional callable passed straight to yt-dlp's own
    'progress_hooks' list, e.g. to update a status label with % downloaded.

    Note: this pulls real audio from YouTube. Only use it for videos you
    have the right to download (your own uploads, royalty-free/Creative
    Commons tracks, etc.) - downloading copyrighted music this way can
    violate YouTube's Terms of Service.
    """
    if not song_name or not song_name.strip():
        raise ValueError("Enter a song name to search for first.")

    search_query = f"ytsearch1:{song_name}"

    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = os.path.join(temp_dir, "yt_audio.%(ext)s")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'noplaylist': True,
            'noprogress': progress_hook is None,
        }
        if progress_hook is not None:
            ydl_opts['progress_hooks'] = [progress_hook]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=True)

        # extract_info on a search query returns a playlist-like result with
        # the matched video(s) under "entries" - grab the top hit's title
        # for status/labeling purposes.
        entries = info.get("entries") if isinstance(info, dict) else None
        title = entries[0].get("title") if entries else info.get("title", song_name)

        wav_path = os.path.join(temp_dir, "yt_audio.wav")
        if not os.path.exists(wav_path):
            raise RuntimeError("Download finished but no audio file was produced.")

        sr, data = load_wav(wav_path)
        return sr, data, title


def fetch_from_archive_org(song_name: str, progress_hook=None):
    """
    Searches the Internet Archive (archive.org) for `song_name` and downloads
    audio from it - a safe alternative to fetch_youtube_by_song_name() above.
    Archive.org hosts public-domain and openly-licensed media specifically
    for free download/reuse, so unlike scraping YouTube, this doesn't touch
    anyone's Terms of Service; you're using the library the way it's meant
    to be used. (It's still on you to pick items that are actually free to
    use - search results can include copyrighted uploads too - but the
    *mechanism* itself is fully sanctioned, unlike stream-ripping.)

    Returns (sample_rate, data, title) - same shape as
    fetch_youtube_by_song_name(), so it's a drop-in replacement.

    progress_hook: optional callable, called with a dict like
    {"status": "searching"/"downloading"/"converting", ...} so the GUI can
    show progress, mirroring yt-dlp's progress_hooks convention.
    """
    if not song_name or not song_name.strip():
        raise ValueError("Enter a song name to search for first.")

    def _notify(status, **extra):
        if progress_hook is not None:
            progress_hook({"status": status, **extra})

    _notify("searching", query=song_name)

    search_resp = requests.get(
        "https://archive.org/advancedsearch.php",
        params={
            "q": f'({song_name}) AND mediatype:(audio)',
            "fl[]": ["identifier", "title"],
            "rows": 5,
            "page": 1,
            "output": "json",
        },
        timeout=20,
    )
    search_resp.raise_for_status()
    docs = search_resp.json().get("response", {}).get("docs", [])
    if not docs:
        raise RuntimeError(f"No Internet Archive results found for '{song_name}'.")

    preferred_exts = (".mp3", ".ogg", ".flac", ".wav", ".m4a")
    last_error = None

    for doc in docs:
        identifier = doc.get("identifier")
        title = doc.get("title") or song_name
        if not identifier:
            continue
        try:
            meta_resp = requests.get(f"https://archive.org/metadata/{identifier}", timeout=20)
            meta_resp.raise_for_status()
            files = meta_resp.json().get("files", [])

            audio_file = next(
                (f for f in files if f.get("name", "").lower().endswith(preferred_exts)),
                None,
            )
            if audio_file is None:
                continue  # this item has no usable audio file - try the next search result

            file_url = f"https://archive.org/download/{identifier}/{audio_file['name']}"
            _notify("downloading", identifier=identifier, title=title)

            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = os.path.join(temp_dir, audio_file["name"])
                with requests.get(file_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as out_f:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            out_f.write(chunk)

                _notify("converting")
                sr, data = load_audio(local_path)
                return sr, data, title

        except Exception as e:
            last_error = e
            continue  # try the next search result rather than failing outright

    raise RuntimeError(
        f"Found Internet Archive results for '{song_name}' but couldn't download "
        f"any of them ({last_error})."
    )


# -------------------------------------------------- voiceprint persistence ----

def fetch_from_jamendo(song_name: str, client_id: str, progress_hook=None):
    """
    Searches Jamendo (a real, licensed catalog of CC-licensed music, with an
    actual public API - not scraping) for `song_name` and downloads the top
    match's audio. Needs a free Jamendo client_id (register at
    https://devportal.jamendo.com). Returns (sr, data, title), same shape as
    fetch_from_archive_org(), so it's a drop-in replacement/alternative.
    """
    if not song_name or not song_name.strip():
        raise ValueError("Enter a song name to search for first.")
    if not client_id or not client_id.strip():
        raise ValueError("Enter your Jamendo client_id first (free at devportal.jamendo.com).")

    def _notify(status, **extra):
        if progress_hook is not None:
            progress_hook({"status": status, **extra})

    _notify("searching", query=song_name)

    resp = requests.get(
        "https://api.jamendo.com/v3.0/tracks/",
        params={
            "client_id": client_id,
            "format": "json",
            "limit": 1,
            "namesearch": song_name,
            "audioformat": "mp32",
        },
        timeout=20,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise RuntimeError(f"No Jamendo results found for '{song_name}'.")

    track = results[0]
    title = track.get("name") or song_name
    file_url = track.get("audio")
    if not file_url:
        raise RuntimeError("Jamendo result had no downloadable audio URL.")

    _notify("downloading", title=title)
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, "jamendo_track.mp3")
        with requests.get(file_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_path, "wb") as out_f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    out_f.write(chunk)

        _notify("converting")
        sr, data = load_audio(local_path)
        return sr, data, title


def resample_naive(data, sr, target_sr):
    """
    Demonstrates ALIASING: downsamples by simply picking every Nth sample,
    with no low-pass ("anti-aliasing") filter beforehand. If target_sr is
    less than 2x the signal's highest frequency (the Nyquist rate), high
    frequencies "fold back" into the wrong, lower frequencies instead of
    being removed - that's what produces the harsh, distorted sound.

    Returns (new_data, new_sr). new_sr == target_sr (fewer samples,
    shorter array) so it plays back at the correct (lower-fidelity) pitch
    and speed.
    """
    if target_sr <= 0 or target_sr > sr:
        raise ValueError("Target rate must be a positive value below the original sample rate.")
    step = sr / target_sr
    indices = np.round(np.arange(0, len(data), step)).astype(int)
    indices = indices[indices < len(data)]
    return data[indices], target_sr


def resample_proper(data, sr, target_sr):
    """
    Demonstrates CORRECT downsampling: uses an FFT-based resampler
    (band-limited by construction) that discards frequency content above
    the new Nyquist limit before reducing the sample count, so no
    aliasing occurs - just a genuine loss of high-frequency detail.

    Returns (new_data, new_sr), same shape/contract as resample_naive()
    so the two can be A/B compared directly.
    """
    if target_sr <= 0 or target_sr > sr:
        raise ValueError("Target rate must be a positive value below the original sample rate.")
    n_target = int(round(len(data) * target_sr / sr))
    new_data = scipy_signal_resample(data, max(n_target, 1))
    return new_data, target_sr


def upsample_to(data, sr, target_sr):
    """
    Upsamples back to `target_sr` (typically the original project sample
    rate) purely so a downsampled clip can be played back / plotted
    alongside the original without changing the audio backend's rate.
    Does not add back any real information lost during downsampling.
    """
    if target_sr == sr:
        return data, sr
    n_target = int(round(len(data) * target_sr / sr))
    new_data = scipy_signal_resample(data, max(n_target, 1))
    return new_data, target_sr


def save_voiceprints(path, enrolled_speakers):
    """Saves the {name: mfcc_vector} enrollment dict to a JSON file so
    registered voices survive between app runs. numpy arrays aren't
    JSON-serializable directly, so each vector is converted to a plain list."""
    serializable = {name: np.asarray(vec).tolist() for name, vec in enrolled_speakers.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def load_voiceprints(path):
    """Loads a {name: mfcc_vector} enrollment dict previously saved with
    save_voiceprints(). Returns an empty dict if the file doesn't exist yet
    (e.g. first run) rather than raising."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {name: np.array(vec) for name, vec in raw.items()}