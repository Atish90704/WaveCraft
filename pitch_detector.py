import math
import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def frequency_to_note(frequency: float):
    """Converts a frequency in Hz to its closest musical note and tuning offset."""
    if not frequency or frequency <= 0:
        return None

    # Calculate MIDI note number relative to A4 (440Hz = MIDI 69)
    note_num = 12 * math.log2(frequency / 440.0) + 69
    rounded_note = round(note_num)

    note_index = int((rounded_note % 12 + 12) % 12)
    octave = int(rounded_note // 12) - 1
    note_name = NOTE_NAMES[note_index]

    # Cents offset (-50 flat to +50 sharp)
    cents = round((note_num - rounded_note) * 100)

    return {
        "note": note_name,
        "octave": octave,
        "full_note": f"{note_name}{octave}",
        "cents": cents # Positive = Sharp, Negative = Flat
    }

def detect_pitch_from_fft(fft_data, sample_rate: int, fft_size: int, min_threshold: float = 15.0):
    """
    Finds the dominant frequency peak from FFT magnitude array.
    :param fft_data: numpy array or list of magnitude values
    """
    if len(fft_data) == 0:
        return {"frequency": 0.0, "note_info": None, "is_silent": True}

    fft_array = np.asarray(fft_data)
    max_val = np.max(fft_array)
    max_index = np.argmax(fft_array)

    # Threshold check to ignore silence / ambient noise
    if max_val < min_threshold or max_index == 0:
        return {"frequency": 0.0, "note_info": None, "is_silent": True}

    dominant_frequency = (max_index * sample_rate) / fft_size
    note_info = frequency_to_note(dominant_frequency)

    return {
        "frequency": round(dominant_frequency, 1),
        "note_info": note_info,
        "is_silent": False
    }