"""
Lightweight sound effects for duel mode -- no external audio files. Each
effect is a short, procedurally-generated PCM buffer (sine/square tones and
frequency sweeps, mixed with a little noise for percussive impacts, shaped
by a simple decay envelope), built once at import time.

Playback prefers `winsound` (Windows standard library -- no pip install, no
C compiler needed) and falls back to `simpleaudio` if that's unavailable
(e.g. on Mac/Linux) and installed. If neither works, playback silently
becomes a no-op instead of crashing the game -- sound is a nice-to-have
layered on top of the gameplay, never a requirement for it to run.
"""

import io
import wave

import numpy as np

try:
    import winsound
    _BACKEND = "winsound"
except ImportError:
    winsound = None
    try:
        import simpleaudio as sa  # type: ignore[import-not-found]
        _BACKEND = "simpleaudio"
    except Exception:
        sa = None
        _BACKEND = None

SAMPLE_RATE = 44100


def _envelope(n, attack=0.01, decay=0.15):
    """Quick fade-in, then exponential decay -- avoids clicky edges and
    gives every effect a natural-feeling tail."""
    t = np.linspace(0, 1, n)
    env = np.ones(n)
    a_n = int(n * attack)
    if a_n > 0:
        env[:a_n] = np.linspace(0, 1, a_n)
    env *= np.exp(-t / max(decay, 1e-3))
    return env


def _tone(freq, duration, wave="sine", noise_mix=0.0, decay=0.15):
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    if wave == "square":
        signal = np.sign(np.sin(2 * np.pi * freq * t))
    else:
        signal = np.sin(2 * np.pi * freq * t)
    if noise_mix > 0:
        noise = np.random.uniform(-1, 1, n)
        signal = (1 - noise_mix) * signal + noise_mix * noise
    signal *= _envelope(n, decay=decay)
    return signal


def _sweep(freq_start, freq_end, duration, noise_mix=0.0):
    """A tone that slides from one frequency to another -- used for the
    punch 'thud' (high-to-low) and the knockdown 'drop' (longer, lower)."""
    n = int(SAMPLE_RATE * duration)
    freq = np.linspace(freq_start, freq_end, n)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    signal = np.sin(phase)
    if noise_mix > 0:
        noise = np.random.uniform(-1, 1, n)
        signal = (1 - noise_mix) * signal + noise_mix * noise
    signal *= _envelope(n, decay=duration * 0.6)
    return signal


def _to_pcm16(signal):
    signal = np.clip(signal, -1.0, 1.0)
    return (signal * 32767).astype(np.int16)


def _pcm_to_wav_bytes(pcm):
    """winsound.PlaySound(..., SND_MEMORY) needs a real WAV file in memory
    (header included), not a bare PCM buffer."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _build_effects():
    return {
        "punch": _to_pcm16(_sweep(220, 80, 0.12, noise_mix=0.5) * 0.9),
        "block": _to_pcm16(_tone(700, 0.06, wave="square", noise_mix=0.2, decay=0.05) * 0.6),
        "knockdown": _to_pcm16(_sweep(160, 40, 0.5, noise_mix=0.3) * 0.9),
        "ko": _to_pcm16(np.concatenate([
            _tone(440, 0.15, wave="square", decay=0.12),
            _tone(330, 0.15, wave="square", decay=0.12),
            _tone(220, 0.35, wave="square", decay=0.3),
        ]) * 0.7),
        "recovery_hit": _to_pcm16(_tone(880, 0.07, decay=0.06) * 0.5),
        "recovered": _to_pcm16(np.concatenate([
            _tone(523, 0.08, decay=0.07),
            _tone(659, 0.08, decay=0.07),
            _tone(784, 0.18, decay=0.15),
        ]) * 0.6),
    }


_EFFECTS = _build_effects() if _BACKEND else {}
_WAV_CACHE = {name: _pcm_to_wav_bytes(pcm) for name, pcm in _EFFECTS.items()} if _BACKEND == "winsound" else {}
_warned = False


def play(name):
    """Fire-and-forget playback of a named effect. Never raises -- a
    missing or misbehaving audio backend should never take the game down
    with it."""
    global _warned
    if not _BACKEND:
        if not _warned:
            print("Sound effects disabled: no usable audio backend found "
                  "(winsound is Windows-only; on Mac/Linux, `pip install simpleaudio`).")
            _warned = True
        return
    try:
        if _BACKEND == "winsound":
            wav_bytes = _WAV_CACHE.get(name)
            if wav_bytes is None:
                return
            winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_ASYNC)
        elif _BACKEND == "simpleaudio":
            pcm = _EFFECTS.get(name)
            if pcm is None:
                return
            sa.play_buffer(pcm, 1, 2, SAMPLE_RATE)
    except Exception:
        pass