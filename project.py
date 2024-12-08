# Mustafa Ali
# CSCI 411
# Fast Fourier Transform
# Project

import numpy as np
import matplotlib.pyplot as plt
import wave
import os
import argparse
from scipy.fft import fft
from scipy.signal import resample


# Function to pad signal to the next power of 2
def pad_to_next_power_of_2(x):
    n = len(x)
    next_power_of_2 = 1 << (n - 1).bit_length()
    return np.pad(x, (0, next_power_of_2 - n), mode='constant')


# Load the audio file as a numpy array
def load_wave(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        params = wav_file.getparams()
        num_channels, sampwidth, framerate, num_frames = params[:4]

        # Read frames and convert to numpy array
        raw_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        # Use left channel if stereo
        if num_channels == 2:
            audio_data = audio_data[::2]

        return audio_data, framerate


# Implemented, but not tested. Reason: too slow
def dft(x):
    new_rate = 1000  # Target sample rate (In the report, you will see why I chose 1000)
    N = int(len(x) * new_rate / len(x))  # Resample length adjustment
    x_resampled = resample(x, N)  # Resample the signal to the new length

    n = np.arange(N)
    k = n.reshape((N, 1))

    e = np.exp(-2j * np.pi * k * n / N)
    X_k = np.zeros_like(k, dtype=np.complex128)

    for row in n:
        for col in n:
            X_k[row] += e[row, col] * x_resampled[col]
    return X_k

# Implemented, but not tested. Reason: too slow
def fft_recursive(x):
    n = len(x)
    if n == 1:
        return x
    even = fft(x[::2])
    odd = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(n) / n)
    return np.concatenate([even + factor[:n // 2] * odd, even - factor[:n // 2] * odd])


# Fast Fourier Transform (FFT) implementation
def fft_vector(x):
    if len(x) == 0 or np.all(x == 0):  # Check if the signal is empty or all zeros
        return np.zeros_like(x)

    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("must be a power of 2")

    N_min = min(N, 2)

    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        terms = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + terms * X_odd, X_even - terms * X_odd])

    return X.ravel()


# Downsample the signal to the target sample rate (used for sample_rates folder)
# We are downsampling a WAV file from 48000 Hertz
def downsample_signal(signal, original_rate, target_rate):
    # Resample the signal to the target sample rate
    num_samples = int(len(signal) * target_rate / original_rate)
    downsampled_signal = resample(signal, num_samples)
    return downsampled_signal


# Function to plot time and frequency domains
def plot_time_freq(file_path, signal, sr, target_sr):
    N = len(signal)
    time = np.linspace(0, N / sr, N)

    # Perform FFT
    signal_fft = fft(signal)
    freqs = np.fft.fftfreq(N, d=1/sr)
    signal_fft_magnitude = np.abs(signal_fft)

    # Plot Time Domain
    plt.figure(figsize=(14, 6))

    # Plot time domain
    plt.subplot(1, 2, 1)
    plt.plot(time, signal, color='blue', alpha=0.7)
    plt.title(f"Time Domain: {file_path} (Sample Rate: {target_sr} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot frequency domain
    plt.subplot(1, 2, 2)
    plt.plot(freqs[:N//2], signal_fft_magnitude[:N//2], color='orange', alpha=0.7)
    plt.title(f"Frequency Domain: {file_path} (Sample Rate: {target_sr} Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Function to process the files in the input directory
def process_files(input_dir, sample_rate_plot=False, dont_plot=False):
    target_sample_rates = [8000, 11025, 16000, 22050, 44100, 48000]

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing {file_name}...")

            # Load the audio file
            signal, original_sr = load_wave(file_path)

            if sample_rate_plot and not dont_plot:
                # Downsample and plot for each target sample rate
                for target_sr in target_sample_rates:
                    downsampled_signal = downsample_signal(signal, original_sr, target_sr)
                    plot_time_freq(file_path, downsampled_signal, target_sr, target_sr)


# Function to process and store results for FFT testing
def process_and_store(test_case, expected_output_file, label, use_fft):
    # Load the expected output file
    with open(expected_output_file, 'r') as file:
        expected_output = file.read().strip()

    try:
        expected_output = expected_output.strip("[]")
        expected_output_array = np.fromstring(expected_output, sep=' ')
        if expected_output_array.size == 0:
            print("Expected output is empty or not correctly formatted.")
            return None, None, False, None
    except ValueError as e:
        print(f"Error parsing expected output from {expected_output_file}: {e}")
        return None, None, False, None

    signal, sr = load_wave(test_case)

    # Check if the signal is all zeros
    if np.all(signal == 0):
        print(signal)
        result_match = np.allclose([0], expected_output_array, atol=1e-6)
        return [0], [0], result_match, sr

    signal = signal / np.max(np.abs(signal))  # Normalize signal
    if np.any(np.isnan(signal)):
        print("Signal contains NaNs after normalization")
    signal = pad_to_next_power_of_2(signal)  # Pad signal to next power of 2

    if use_fft:
        fft_result_custom = fft_vector(signal)
    else:
        fft_result_custom = dft(signal)

    fft_result_numpy = np.fft.fft(signal)

    fft_magnitude_custom = np.abs(fft_result_custom)
    fft_magnitude_numpy = np.abs(fft_result_numpy)

    # Check for NaNs in the results
    if np.any(np.isnan(fft_magnitude_numpy)):
        print("NaN values found in FFT magnitude (NumPy)")

    print("FFT Magnitude (Custom):", fft_magnitude_custom)
    print("FFT Magnitude (NumPy):", fft_magnitude_numpy)

    result_match = np.allclose(fft_magnitude_custom, expected_output_array, atol=1e-6)
    return fft_magnitude_custom, fft_magnitude_numpy, result_match, sr




# Function to plot results with frequency (Hz) on the x-axis
def plot_results(results):
    for label, fft_magnitude_custom, fft_magnitude_numpy, sr in results:
        N = len(fft_magnitude_custom)
        freqs = np.fft.fftfreq(N, d=1/sr)  # Calculate frequency bins

        plt.figure(figsize=(14, 6))

        # Plot side-by-side
        plt.subplot(1, 2, 1)
        plt.plot(freqs[:N//2], np.abs(fft_magnitude_custom)[:N//2], label="Custom FFT Magnitude", color='blue', alpha=0.7)
        plt.title(f"Custom FFT Magnitude for {label}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(freqs[:N//2], np.abs(fft_magnitude_numpy)[:N//2], label="NumPy FFT Magnitude", color='orange', alpha=0.7)
        plt.title(f"NumPy FFT Magnitude for {label}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Run tests on the files in the test_input directory
def run_tests(input_dir, output_dir, use_fft=True, dont_plot=False):
    results = []  # Store results for plotting later

    # Loop through all wav files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            test_case = os.path.join(input_dir, file_name)
            label = file_name.split('.')[0]  # Use file name without extension as label
            expected_output_file = os.path.join(output_dir, f"{label}_output.out")  # Output file path

            if not os.path.exists(expected_output_file):
                print(f"Expected output file {expected_output_file} does not exist.")
                continue

            print(f"Processing {file_name}... \n")

            # Process the test case and collect results
            fft_result_custom, fft_result_numpy, result_match, sr = process_and_store(
                test_case, expected_output_file, label, use_fft
            )

            # Save the results for plotting
            if fft_result_custom is not None and fft_result_numpy is not None:
                results.append((label, fft_result_custom, fft_result_numpy, sr))

            if result_match:
                print(f"{label}: Test passed. \n")
            else:
                print(f"{label}: Test failed. \n")

    # Plot results after all test cases (if not using --dont_plot)
    if not dont_plot:
        plot_results(results)


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="FFT Testing and Visualization")
    parser.add_argument("input_dir", type=str, help="Directory containing input WAV files")
    parser.add_argument("output_dir", type=str, help="Directory containing expected output files")
    parser.add_argument("--sample_rate_plot", action="store_true", help="Enable sample rate plot visualization")
    parser.add_argument("--dont_plot", action="store_true", help="Disable plotting and only compute test cases")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.sample_rate_plot:
        process_files(args.input_dir, sample_rate_plot=True, dont_plot=args.dont_plot)
    else:
        run_tests(args.input_dir, args.output_dir, dont_plot=args.dont_plot)


if __name__ == "__main__":
    main()
