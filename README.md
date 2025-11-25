# Synth-Studio

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)
![Dependencies](https://img.shields.io/badge/dependencies-NumPy%20%7C%20SciPy-orange.svg)

## Project Abstract

**Synth-Studio** is a real-time modular synthesizer and sequencer implemented entirely in Python without reliance on pre-built audio engines. The system leverages low-level digital signal processing principles to generate polyphonic audio synthesis through mathematical waveform generation, stateful filtering, and time-domain effects processing. Built on `sounddevice` for ASIO-level audio stream management and `numpy` for vectorized DSP operations, the architecture provides deterministic real-time audio rendering with thread-safe voice management and sample-accurate timing resolution.

The project demonstrates practical implementation of fundamental audio synthesis concepts including additive/subtractive waveform generation, IIR filter design for noise shaping, ring-buffer delay implementation, and JSON-based event sequencing for polyphonic composition playback.

---
<img width="751" height="474" alt="demo0" src="https://github.com/user-attachments/assets/a2b1685f-caad-4007-a8b6-50e8388ee496" />

---

## Repository Structure

```
Synth-Studio/
├── docs/
│   └── composition_manual.pdf  # Comprehensive guide on creating .synth files
├── songs/
│   ├── lofi.synth              # Example Low-Fidelity composition
│   └── neon.synth              # Example Retro-style composition
├── main.py                     # Entry point and core DSP engine
├── requirements.txt            # Python dependencies
└── README.md
```

**Directory Descriptions:**

- **`docs/`**: Technical documentation including the `.synth` file format specification and composition guidelines.
- **`songs/`**: Pre-configured composition files demonstrating synthesis capabilities and sequencing patterns.
- **`main.py`**: Primary executable containing the audio callback engine, DSP chain implementation, voice allocation logic, and user interface.
- **`requirements.txt`**: Pinned dependency manifest for reproducible environment configuration.

---

## Architecture & Signal Flow

The audio rendering pipeline follows a modular signal flow architecture:

### 1. **Voice Generation (`SynthVoice` Objects)**
Each active note instantiates a `SynthVoice` object responsible for:
- **Waveform Synthesis**: Per-sample generation using NumPy-based mathematical oscillators (sine, triangle, sawtooth, square wave implementations via phase accumulation and trigonometric/piecewise functions).
- **ADSR Envelope**: Amplitude modulation through discrete attack/decay/sustain/release state machine with linear interpolation.
- **Pink Noise Generation**: Stateful IIR filter application using `scipy.signal.lfilter` with pole-zero placement coefficients to achieve 1/f spectral distribution from white noise source.

### 2. **Voice Summing & Polyphony Management**
Active voices are summed in the audio callback with:
- **Thread-Safe Access**: `threading.Lock` guards voice list mutations during real-time audio thread execution.
- **Dynamic Allocation**: Voices are spawned/released based on sequencer events with automatic cleanup of expired envelopes.

### 3. **Effects Chain Processing**
Post-summation signal passes through cascaded effects:
- **Low-Pass Filter**: First-order IIR implementation with configurable cutoff frequency for frequency domain attenuation.
- **Delay Effect**: Circular buffer (ring buffer) implementation providing time-domain echo with feedback coefficient and wet/dry mixing.

### 4. **Audio Output**
Processed samples are written to the `sounddevice` output stream buffer with:
- **Sample Rate**: 44.1 kHz standard
- **Bit Depth**: 32-bit float internal processing
- **Latency**: Configurable block size for latency/stability tradeoff

**Signal Path Diagram:**
```
JSON Sequencer → [Note Events] → SynthVoice Generation → Polyphonic Summation
                                                              ↓
                                                    Effects Chain (LPF → Delay)
                                                              ↓
                                                    sounddevice Output Buffer
```

---

## Installation & Usage

### Prerequisites

- **Python**: Version 3.10 or higher
- **Operating System**: Cross-platform (Windows, macOS, Linux with ALSA/JACK)
- **Audio Interface**: System audio output device

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/Synth-Studio.git
cd Synth-Studio
pip install -r requirements.txt
```

**Dependencies:**
- `numpy`: Vectorized numerical computation for DSP operations
- `scipy`: Signal processing library for filter design
- `sounddevice`: Cross-platform audio I/O interface

### Execution

Launch the synthesizer engine:

```bash
python main.py
```

### Operation

Upon execution, the application presents an interactive interface with the following controls:

1. **File Loading**: Use the interface to navigate to the `songs/` directory and select a `.synth` composition file.
2. **Playback Control**: Start/stop sequencer playback with transport controls.
3. **Real-Time Parameter Adjustment**: Modify synthesis parameters (waveform type, filter cutoff, delay time/feedback) during playback.
4. **Voice Monitoring**: Observe active voice count and CPU utilization metrics in real-time.

**Example Workflow:**
```bash
# Start the application
python main.py

# In the interface:
# 1. Click "Load File" → Select "songs/lofi.synth"
# 2. Adjust BPM if desired (default: file-specified tempo)
# 3. Click "Play" to begin sequencer playback
# 4. Modify filter/delay parameters during playback to hear real-time changes
```

---

## File Format Specification (.synth)

The `.synth` file format is a **JSON-structured** composition descriptor that defines temporal sequencing and synthesis parameters.

### Schema Structure

```json
{
  "bpm": <integer>,
  "loop": <boolean>,
  "notes": [
    {
      "time": <float>,
      "frequency": <float>,
      "duration": <float>,
      "waveform": <string>,
      "amplitude": <float>
    }
  ]
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| **`bpm`** | Integer | Tempo in beats per minute (affects time-to-sample conversion) |
| **`loop`** | Boolean | Enables automatic composition restart upon completion |
| **`notes`** | Array | Ordered collection of note event objects |
| **`time`** | Float | Event trigger time in beats (supports fractional timing for sub-beat resolution) |
| **`frequency`** | Float | Fundamental frequency in Hertz (A4 = 440 Hz standard) |
| **`duration`** | Float | Note sustain duration in beats (controls ADSR release trigger point) |
| **`waveform`** | String | Oscillator type: `"sine"`, `"triangle"`, `"sawtooth"`, `"square"`, `"pink_noise"` |
| **`amplitude`** | Float | Normalized amplitude (0.0 to 1.0 range, scales ADSR envelope output) |

### Example Composition

```json
{
  "bpm": 90,
  "loop": true,
  "notes": [
    {
      "time": 0.0,
      "frequency": 261.63,
      "duration": 1.0,
      "waveform": "sine",
      "amplitude": 0.7
    },
    {
      "time": 1.0,
      "frequency": 329.63,
      "duration": 0.5,
      "waveform": "sawtooth",
      "amplitude": 0.6
    }
  ]
}
```

**Technical Notes:**
- **Polyphony**: Multiple simultaneous notes at identical `time` values are supported through voice pooling.
- **Timing Resolution**: Float-based timing allows for sample-accurate event placement independent of beat quantization.
- **Frequency Precision**: Direct frequency specification enables microtonal composition without MIDI note number constraints.

---

## Technical Implementation Details

### Waveform Generation Algorithms

All waveforms are generated through phase accumulation and mathematical transformation:

- **Sine Wave**: `np.sin(2 * np.pi * phase)`
- **Triangle Wave**: Piecewise linear approximation using phase-dependent slope calculation
- **Sawtooth Wave**: Linear ramp with modulo wraparound: `2 * (phase % 1.0) - 1`
- **Square Wave**: Sign function with duty cycle threshold: `np.sign(np.sin(2 * np.pi * phase))`
- **Pink Noise**: White noise passed through cascaded first-order IIR filters with coefficients derived from Voss-McCartney algorithm

### Thread Safety

The audio callback operates in a dedicated real-time thread. All shared state modifications (voice list, sequencer state) are protected by `threading.Lock` primitives to prevent race conditions and ensure deterministic audio rendering.

### Performance Characteristics

- **Latency**: Typical round-trip latency of 10-20ms depending on buffer configuration
- **Polyphony**: Tested stable with 64+ simultaneous voices on modern hardware
- **CPU Usage**: Approximately 5-15% single-core utilization at 44.1kHz sample rate with moderate polyphony

---

## License

This project is released under the **MIT License**. See `LICENSE` file for full terms.

---

## Contributing

Contributions addressing DSP optimization, additional waveform algorithms, or effect implementations are welcome. Please ensure all submissions maintain real-time performance characteristics and include appropriate unit tests for signal processing functions.
