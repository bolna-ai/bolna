#!/usr/bin/env python3
"""
Prepare optimized ambient noise files for telephony use.
Converts to 8kHz, mono, normalized levels.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.io import wavfile
from scipy import signal
import numpy as np

def prepare_file(input_path, output_path, target_rate=8000, target_peak=20000):
    """Convert audio to telephony-ready format"""
    print(f"\nProcessing: {os.path.basename(input_path)}")
    print("-" * 60)

    # Read original
    rate, data = wavfile.read(input_path)
    print(f"  Input:  {rate}Hz, ", end="")

    # Convert stereo to mono
    if len(data.shape) == 2:
        print(f"stereo → mono")
        data = data.mean(axis=1).astype(np.int16)
    else:
        print(f"mono")

    # Resample to 8kHz if needed
    if rate != target_rate:
        print(f"  Resample: {rate}Hz → {target_rate}Hz")
        num_samples = int(len(data) * target_rate / rate)
        data = signal.resample(data, num_samples)

    # Normalize to target peak level
    current_peak = np.abs(data).max()
    if current_peak > 0:
        normalization_factor = target_peak / current_peak
        data = data * normalization_factor
        print(f"  Normalize: peak {current_peak:.0f} → {target_peak}")

    # Ensure int16 format
    data = np.clip(data, -32768, 32767).astype(np.int16)

    # Write output
    wavfile.write(output_path, target_rate, data)

    # Verify
    final_rate, final_data = wavfile.read(output_path)
    final_peak = np.abs(final_data).max()
    final_rms = np.sqrt(np.mean(final_data.astype(float)**2))
    duration = len(final_data) / final_rate

    print(f"  Output: {final_rate}Hz, {duration:.1f}s, peak={final_peak:.0f}, RMS={final_rms:.1f}")
    print(f"  ✓ Saved: {output_path}")


def main():
    source_dir = "local_setup/presets/ambient_noise"
    output_dir = "presets/ambient_noise"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Preparing Telephony-Ready Ambient Noise Files")
    print("=" * 60)

    files = {
        "call-center.wav": "call-center.wav",
        "coffee-shop.wav": "coffee-shop.wav",
        "office-ambience.wav": "office-ambience.wav"
    }

    for source_name, output_name in files.items():
        source_path = os.path.join(source_dir, source_name)
        output_path = os.path.join(output_dir, output_name)

        if os.path.exists(source_path):
            prepare_file(source_path, output_path, target_rate=8000, target_peak=20000)
        else:
            print(f"\n⚠ Skipping {source_name} - file not found")

    print("\n" + "=" * 60)
    print("✓ Telephony-ready ambient files created in presets/ambient_noise/")
    print("  - 8kHz sample rate (telephony standard)")
    print("  - Mono (single channel)")
    print("  - Normalized to ~20k peak (good balance)")
    print("  - At 30% volume: ~6000 peak (subtle background)")
    print("  - At 15% volume: ~3000 peak (very subtle)")
    print("=" * 60)


if __name__ == "__main__":
    main()
