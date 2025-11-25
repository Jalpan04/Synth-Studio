"""
Synth Studio v3.3 — Cleaned, production-ready single-file script
Features:
 - real-time synth voices with multiple waveforms
 - ADSR-style amplitude handling
 - low-pass filter and delay effect
 - JSON sequencer loader and playback
 - record to WAV

Notes:
 - Keep sample rate and block size unchanged for predictable latency
 - Run on a system with sounddevice, scipy, pygame and tkinter available
"""

import json
import os
import time
import threading
import colorsys
from typing import Dict, List

import numpy as np
import pygame
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog
from scipy import signal
from scipy.io.wavfile import write as write_wav

# -------------------- Configuration --------------------
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
WIDTH, HEIGHT = 1000, 600

# UI colors
C_BG = (10, 10, 12)
C_GRID = (30, 30, 35)
C_TEXT = (220, 220, 220)
C_TEXT_DIM = (100, 100, 110)
C_ACCENT = (50, 200, 150)
C_REC = (255, 50, 50)
C_ALERT = (255, 100, 100)

# Waveform color mapping (HSV hue values)
WAVE_HUES = {
    "sine": 0.6, "triangle": 0.35, "square": 0.75, "sawtooth": 0.05,
    "noise": 0.95, "bell": 0.14, "organ": 0.55, "trance": 0.50,
    "scifi": 0.8, "crush": 0.25,
}

# Note table for octave 4 reference
NOTES = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "G": 392.00,
    "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88,
}

# Pink noise filter coefficients (approximating -3 dB / octave)
PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
PINK_A = [1.0, -2.494956002, 2.017265875, -0.522189400]

# -------------------- Utilities --------------------

def parse_note(note_str: str) -> float:
    """Parse a note like 'C4' or 'A#3' into frequency in Hz.
    Returns 0.0 for invalid input.
    """
    if not note_str:
        return 0.0
    try:
        note = note_str[:-1]
        octave = int(note_str[-1])
        base = NOTES.get(note, 0.0)
        return base * (2 ** (octave - 4))
    except Exception:
        return 0.0


# -------------------- Effects --------------------
class LowPassFilter:
    """Very lightweight one-pole-style smoothing implemented as a simple
    exponential smoother across a whole audio buffer.
    """

    def __init__(self) -> None:
        self.cutoff = 1.0
        self._last = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        alpha = self.cutoff
        if alpha >= 0.99:
            return audio
        output = np.empty_like(audio)
        current = self._last
        for i, x in enumerate(audio):
            current += (x - current) * alpha
            output[i] = current
        self._last = current
        return output


class DelayEffect:
    """Circular buffer delay with simple feedback/decay.
    Toggle `active` to enable or disable.
    """

    def __init__(self, delay_seconds: float = 0.4, decay: float = 0.5) -> None:
        self.buffer_size = int(SAMPLE_RATE * delay_seconds)
        self.buffer = np.zeros(self.buffer_size, dtype=float)
        self.cursor = 0
        self.decay = decay
        self.active = False

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.active or len(audio) > self.buffer_size:
            return audio
        idx = (self.cursor + np.arange(len(audio))) % self.buffer_size
        delayed = self.buffer[idx]
        out = audio + delayed * self.decay
        self.buffer[idx] = out
        self.cursor = (self.cursor + len(audio)) % self.buffer_size
        return out


# -------------------- Synth voice --------------------
class SynthVoice:
    """Single voice instance producing one waveform and applying
    a simple ADSR-like amplitude behaviour with a release curve.
    """

    def __init__(self, freq: float, wave_type: str, vol: float = 1.0) -> None:
        self.freq = max(1.0, float(freq))
        self.wave_type = wave_type
        self.phase = 0.0
        self.active = True

        # Attack/release controls
        self.current_amp = 0.0
        self.target_amp = float(vol)
        self.releasing = False
        self.release_factor = 0.92
        self.attack_speed = 0.05

        # Pink noise filter state
        self._pink_zi = signal.lfilter_zi(PINK_B, PINK_A)

        # Visual line color
        hue = WAVE_HUES.get(wave_type, 0.5)
        sat = 0.6 if wave_type == "noise" else 0.8
        val = min(1.0, 0.5 + (self.freq / 800.0))
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        self.color = (int(r * 255), int(g * 255), int(b * 255))

        # Last generated block for visualization
        self.vis_data = np.zeros(BLOCK_SIZE, dtype=float)

    def _generate_wave(self, num_samples: int) -> np.ndarray:
        t = np.linspace(0, num_samples / SAMPLE_RATE, num_samples, endpoint=False)
        phase_vec = self.phase + (2 * np.pi * self.freq * t)

        wt = self.wave_type
        if wt == "sine":
            return np.sin(phase_vec)

        if wt == "triangle":
            saw = 2 * (phase_vec / (2 * np.pi) - np.floor(phase_vec / (2 * np.pi) + 0.5))
            return 2 * np.abs(saw) - 1

        if wt == "square":
            return np.sign(np.sin(phase_vec))

        if wt == "sawtooth":
            return 2 * (phase_vec / (2 * np.pi) - np.floor(phase_vec / (2 * np.pi) + 0.5))

        if wt == "noise":
            white = np.random.uniform(-1.0, 1.0, num_samples)
            pink, self._pink_zi = signal.lfilter(PINK_B, PINK_A, white, zi=self._pink_zi)
            return pink * 3.0

        if wt == "bell":
            mod_phase = phase_vec * 2.0
            modulator = 2.5 * np.sin(mod_phase)
            return np.sin(phase_vec + modulator) * 0.8

        if wt == "organ":
            w1 = np.sin(phase_vec)
            w2 = np.sin(phase_vec * 2.0) * 0.5
            w3 = np.sin(phase_vec * 1.5) * 0.3
            return (w1 + w2 + w3) * 0.6

        if wt == "trance":
            def _saw(p):
                return 2 * (p / (2 * np.pi) - np.floor(p / (2 * np.pi) + 0.5))

            w1 = _saw(phase_vec)
            w2 = _saw(phase_vec * 0.99)
            w3 = _saw(phase_vec * 1.01)
            return (w1 + w2 + w3) * 0.35

        if wt == "scifi":
            modulator = 5.0 * np.sin(phase_vec * 0.5)
            return np.sin(phase_vec + modulator)

        if wt == "crush":
            raw = np.sin(phase_vec)
            steps = 4
            return np.round(raw * steps) / steps

        return np.zeros(num_samples)

    def get_audio(self, num_samples: int) -> np.ndarray:
        if not self.active:
            return np.zeros(num_samples)

        wave = self._generate_wave(num_samples)

        # Envelope handling
        if self.releasing:
            fade = np.power(self.release_factor, np.arange(num_samples))
            wave *= fade * self.current_amp
            self.current_amp *= (self.release_factor ** num_samples)
            if self.current_amp < 0.001:
                self.active = False
        elif self.current_amp < self.target_amp:
            next_amp = min(self.target_amp, self.current_amp + self.attack_speed)
            fade_in = np.linspace(self.current_amp, next_amp, num_samples)
            wave *= fade_in
            self.current_amp = next_amp
        else:
            wave *= self.target_amp

        self.phase += 2 * np.pi * self.freq * (num_samples / SAMPLE_RATE)
        self.phase %= (2 * np.pi * 100)

        self.vis_data = wave
        return wave * 0.15


# -------------------- Audio engine --------------------
class AudioEngine:
    def __init__(self) -> None:
        self.voices: Dict[str, SynthVoice] = {}
        self.lock = threading.Lock()
        self.recording = False
        self.record_buffer: List[np.ndarray] = []
        self.master_vol = 0.8

        self.live_wave = "triangle"
        self.live_octave = 0

        self.filter = LowPassFilter()
        self.delay = DelayEffect()

    def note_on(self, ident: str, freq: float, wave: str, vol: float = 1.0) -> None:
        with self.lock:
            self.voices[ident] = SynthVoice(freq, wave, vol)

    def note_off(self, ident: str) -> None:
        with self.lock:
            if ident in self.voices:
                self.voices[ident].releasing = True

    def stop_sequencer_notes(self) -> None:
        with self.lock:
            for key, v in list(self.voices.items()):
                if key.startswith("seq_"):
                    v.releasing = True
                    v.release_factor = 0.5

    def toggle_record(self) -> str:
        self.recording = not self.recording
        if self.recording:
            self.record_buffer = []
            return "REC STARTED"

        if len(self.record_buffer) == 0:
            return "REC EMPTY"

        full_audio = np.concatenate(self.record_buffer)
        mx = np.max(np.abs(full_audio))
        if mx > 0:
            full_audio = full_audio / mx

        filename = f"rec_{int(time.time())}.wav"
        write_wav(filename, SAMPLE_RATE, full_audio.astype(np.float32))
        return f"SAVED: {filename}"

    def callback(self, outdata, frames, time_info, status) -> None:
        mix = np.zeros(frames, dtype=float)
        with self.lock:
            # remove finished voices
            finished = [k for k, v in self.voices.items() if not v.active]
            for k in finished:
                del self.voices[k]

            for v in self.voices.values():
                mix += v.get_audio(frames)

            mix *= self.master_vol
            mix = self.filter.process(mix)
            mix = self.delay.process(mix)

        mix = np.tanh(mix)
        if self.recording:
            self.record_buffer.append(mix.copy())
        outdata[:] = mix.reshape(-1, 1)


# -------------------- Sequencer --------------------
class Sequencer:
    def __init__(self, engine: AudioEngine) -> None:
        self.engine = engine
        self.playing = False
        self.data = None
        self.events: List[dict] = []
        self.play_thread: threading.Thread | None = None

    def load_file_dialog(self) -> str:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            parent=root,
            title="Open Synth File",
            filetypes=[("JSON/Synth", "*.json *.synth"), ("All", "*.*")],
        )
        root.destroy()
        if not path:
            return "No File"

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            self.data = raw
            self.events = []

            if "tracks" in raw:
                for track in raw["tracks"]:
                    trk_wave = track.get("wave", "sine")
                    for n in track.get("notes", []):
                        self.events.append({
                            "t": n["time"],
                            "n": n["note"],
                            "d": n.get("dur", 0.5),
                            "w": n.get("wave", trk_wave),
                            "v": n.get("vol", 0.6),
                        })

            elif "events" in raw:
                self.events = raw["events"]

            self.events.sort(key=lambda e: e["t"])
            loop_mode = "LOOP" if raw.get("loop", False) else "ONCE"
            return f"LOADED: {os.path.basename(path)} [{loop_mode}]"

        except Exception:
            return "ERR: File Format"

    def _playback_loop(self) -> None:
        if not self.data or not self.events:
            self.playing = False
            return

        bpm = self.data.get("bpm", 120)
        beat_dur = 60.0 / bpm
        loop_len = self.data.get("length", 0)
        if loop_len == 0:
            loop_len = max(e["t"] + e.get("d", 0.5) for e in self.events) + 1

        do_loop = bool(self.data.get("loop", False))
        start_time = time.time()
        idx = 0
        active_notes: List[dict] = []

        try:
            while self.playing:
                now = time.time() - start_time
                cur_beat = now / beat_dur

                if cur_beat >= loop_len:
                    if do_loop:
                        start_time += loop_len * beat_dur
                        cur_beat = (time.time() - start_time) / beat_dur
                        idx = 0
                        for note in active_notes:
                            self.engine.note_off(note["id"])
                        active_notes = []
                    else:
                        break

                while idx < len(self.events) and self.events[idx]["t"] <= cur_beat:
                    ev = self.events[idx]
                    freq = parse_note(ev["n"])
                    ident = f"seq_{idx}_{int(time.time() * 100)}"
                    self.engine.note_on(ident, freq, ev.get("w", "sine"), ev.get("v", 0.5))
                    active_notes.append({"id": ident, "off_t": cur_beat + ev.get("d", 0.5)})
                    idx += 1

                remaining = []
                for note in active_notes:
                    if cur_beat >= note["off_t"]:
                        self.engine.note_off(note["id"])
                    else:
                        remaining.append(note)
                active_notes = remaining

                time.sleep(0.005)

        finally:
            self.playing = False
            self.engine.stop_sequencer_notes()

    def toggle(self) -> str:
        if self.playing:
            self.playing = False
            return "SEQ STOPPED"

        if not self.data:
            return "NO DATA"

        self.playing = True
        self.play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.play_thread.start()
        return "SEQ PLAYING"


# -------------------- Main UI --------------------
def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Synth Studio v3.3 | Pink Noise Edition")
    clock = pygame.time.Clock()

    # Fonts with safe fallbacks
    font_s = pygame.font.SysFont("Consolas", 12)
    font_m = pygame.font.SysFont("Consolas", 16)
    font_b = pygame.font.SysFont("Consolas", 24, bold=True)

    engine = AudioEngine()
    sequencer = Sequencer(engine)

    stream = sd.OutputStream(
        channels=1, blocksize=BLOCK_SIZE, samplerate=SAMPLE_RATE, callback=engine.callback
    )
    stream.start()

    key_map = {
        pygame.K_a: "C4", pygame.K_s: "D4", pygame.K_d: "E4", pygame.K_f: "F4",
        pygame.K_g: "G4", pygame.K_h: "A4", pygame.K_j: "B4", pygame.K_k: "C5",
    }

    status_msg = "Ready."
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    if event.key == pygame.K_l:
                        status_msg = sequencer.load_file_dialog()
                    if event.key == pygame.K_TAB:
                        status_msg = sequencer.toggle()
                    if event.key == pygame.K_r:
                        status_msg = engine.toggle_record()

                    if event.key == pygame.K_SPACE:
                        engine.delay.active = not engine.delay.active
                    if event.key == pygame.K_LEFT:
                        engine.filter.cutoff = max(0.05, engine.filter.cutoff - 0.1)
                    if event.key == pygame.K_RIGHT:
                        engine.filter.cutoff = min(1.0, engine.filter.cutoff + 0.1)
                    if event.key == pygame.K_UP:
                        engine.master_vol = min(2.0, engine.master_vol + 0.1)
                    if event.key == pygame.K_DOWN:
                        engine.master_vol = max(0.0, engine.master_vol - 0.1)

                    # Select live waveform
                    if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                                     pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
                                     pygame.K_9, pygame.K_0]:
                        mapping = {
                            pygame.K_1: "sine", pygame.K_2: "triangle", pygame.K_3: "square",
                            pygame.K_4: "sawtooth", pygame.K_5: "noise", pygame.K_6: "bell",
                            pygame.K_7: "organ", pygame.K_8: "trance", pygame.K_9: "scifi",
                            pygame.K_0: "crush",
                        }
                        engine.live_wave = mapping.get(event.key, engine.live_wave)

                    if event.key == pygame.K_z:
                        engine.live_octave -= 1
                    if event.key == pygame.K_x:
                        engine.live_octave += 1

                    if event.key in key_map:
                        freq = parse_note(key_map[event.key]) * (2 ** engine.live_octave)
                        engine.note_on(str(event.key), freq, engine.live_wave)

                if event.type == pygame.KEYUP and event.key in key_map:
                    engine.note_off(str(event.key))

            # Drawing
            screen.fill(C_BG)

            # Header
            pygame.draw.rect(screen, (20, 20, 25), (0, 0, WIDTH, 50))
            if engine.recording:
                pygame.draw.circle(screen, C_REC, (WIDTH - 30, 25), 8)
            screen.blit(font_b.render(f"SYNTH STUDIO v3.3 | {status_msg}", True, C_TEXT), (20, 15))

            # Effects area
            fx_y = 70
            screen.blit(font_m.render(f"VOL: {int(engine.master_vol * 100)}%", True, C_TEXT), (20, fx_y))
            bar_w = 150
            pygame.draw.rect(screen, (40, 40, 45), (150, fx_y + 5, bar_w, 10))
            fill_w = int(engine.filter.cutoff * bar_w)
            f_col = C_ACCENT if engine.filter.cutoff > 0.3 else C_ALERT
            pygame.draw.rect(screen, f_col, (150, fx_y + 5, fill_w, 10))
            screen.blit(font_s.render("FILTER (Arrow L/R)", True, C_TEXT_DIM), (150, fx_y - 15))
            d_col = C_ACCENT if engine.delay.active else C_TEXT_DIM
            screen.blit(font_m.render(f"DELAY [SPACE]: {'ON' if engine.delay.active else 'OFF'}", True, d_col), (330, fx_y))

            # Waveform visualization
            center_y = HEIGHT // 2 + 20
            pygame.draw.line(screen, C_GRID, (0, center_y), (WIDTH, center_y), 1)
            with engine.lock:
                voices = list(engine.voices.values())
            for v in voices:
                if not v.active:
                    continue
                data = v.vis_data
                step = 3
                scale = 100 * v.current_amp
                points = [(i * (WIDTH / len(data)), center_y - (data[i] * scale))
                          for i in range(0, len(data), step)]
                if len(points) > 1:
                    pygame.draw.lines(screen, v.color, False, points, 2)

            # Footer
            pygame.draw.rect(screen, (15, 15, 18), (0, HEIGHT - 70, WIDTH, 70))
            line1 = "1:SINE 2:TRI 3:SQR 4:SAW 5:NOISE 6:BELL 7:ORGN 8:TRANCE 9:SCI 0:CRSH"
            line2 = f"LIVE WAVE: {engine.live_wave.upper()} | OCTAVE: {engine.live_octave} | KEYS: A-K"
            screen.blit(font_s.render(line1, True, C_TEXT_DIM), (20, HEIGHT - 60))
            screen.blit(font_m.render(line2, True, C_ACCENT), (20, HEIGHT - 30))
            screen.blit(font_s.render("[L]LOAD [TAB]PLAY [R]REC", True, C_TEXT), (WIDTH - 250, HEIGHT - 30))

            pygame.display.flip()
            clock.tick(60)

    finally:
        sequencer.playing = False
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        pygame.quit()


if __name__ == "__main__":
    main()
