import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for interactive plots
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, welch

def estimate_cutoff_frequency(data, fs, power_threshold=0.99):
    """Dynamically estimate the low-pass filter cutoff frequency based on noise analysis.
       This function computes the Power Spectral Density (PSD) and determines the frequency
    where the cumulative power exceeds a specified threshold (default: 99%)."""
    f, Pxx = welch(data, fs=fs, nperseg=len(data)//8)  # Compute Power Spectral Density (PSD)

    cumulative_power = np.cumsum(Pxx) / np.sum(Pxx)  # Compute cumulative power distribution
    cutoff_idx = np.where(cumulative_power >= power_threshold)[0][0]  # Find index where power exceeds threshold
    estimated_cutoff = f[cutoff_idx]  # Corresponding frequency
    
    return min(estimated_cutoff, 15)  # Ensure a reasonable cutoff frequency (at least 1 Hz)

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Applies a Butterworth low-pass filter to reduce high-frequency noise."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def plot_peaks(snippet, frequency=None, distance=5, 
               derivative=False, integrate=False, frequency_domain=False, 
               cutoff_hz=None, label="Force Data"):

    from pathlib import Path
    import pandas as pd
    
    # If snippet is a CSV file path, load metadata and data separately
    if isinstance(snippet, (str, Path)):
        try:
            with open(snippet, 'r') as f:
                metadata_lines = [next(f).strip() for _ in range(6)]
            for line in metadata_lines:
                if "Scan Rate:" in line:
                    frequency = int(line.split(":")[1].strip().replace('"', ''))
                    print(f"Extracted frequency from metadata: {frequency} Hz")
                    break
        except Exception as e:
            raise ValueError("Could not extract frequency from metadata.") from e
        
        snippet = pd.read_csv(snippet, skiprows=6)

    # Default fallback if frequency is still not set
    if frequency is None:
        frequency = 1000  # default value
        print(f"Deafult frequency used: {frequency} Hz")

    snippet['Time (seconds)'] = snippet['Sample'] / frequency

    print(f"Initial Distance: {distance:.2f} seconds")
    # Convert `distance` from seconds to samples
    distance_samples = distance * frequency

    # Remove the mean force value once at the beginning
    force_corrected = snippet['AI0 (V)'] - np.mean(snippet['AI0 (V)'])
    # Convert to mN using the sensor's specification
    volts_per_lb = 2.2  # 2200 mV/lb
    newtons_per_lb = 4.44822162
    conversion_factor = (newtons_per_lb / volts_per_lb) * 1000  # mN per Volt
    force_mN = force_corrected * conversion_factor

    # Determine transformation method based on flags
    derivative_signal = None
    integrated_signal = np.cumsum(force_mN) / frequency if integrate else None

    if derivative:
        if frequency_domain:
            # If frequency domain analysis is enabled, apply low-pass filter
            if cutoff_hz is None:
                cutoff_hz = estimate_cutoff_frequency(force_mN, frequency)
                print(f"Estimated cutoff frequency: {cutoff_hz:.2f} Hz")
            smoothed_force = butter_lowpass_filter(force_mN, cutoff_hz, frequency)
            derivative_signal = np.gradient(smoothed_force)
        else:
            # Compute derivative directly without filtering
            derivative_signal = np.gradient(force_mN)
        
    # By default, use force_corrected as signal for peak detection.
    signal = force_mN  
    if derivative:
        signal = derivative_signal
    elif integrate:
        signal = integrated_signal

    # ----- Adaptive Peak Detection Strategy -----
    # Instead of detecting peaks on the signal, detect on the inverted signal
    inverted_signal = -signal  # Invert the signal

    # ----- Determine Threshold from the Inverted Derivative -----
    sorted_inverted = np.sort(inverted_signal)[::-1]  # descending sort
    print(f"Top 5 values of inverted signal (for threshold estimation): {sorted_inverted[:5]}")

    peak_window_samples = int(0.03 * frequency) + 1
    if len(sorted_inverted) > peak_window_samples:
        reference_peak = sorted_inverted[peak_window_samples]
    elif len(sorted_inverted) > 0:
        reference_peak = sorted_inverted[-1]
    else:
        reference_peak = 0

    print(f"Selected reference peak (index {peak_window_samples}): {reference_peak:.5f}")

    height_threshold = 0.35 * reference_peak
    print(f"Height Threshold (35% of reference peak): {height_threshold:.5f}")

    # Initial peak detection using the height threshold only
    initial_peaks, properties = find_peaks(inverted_signal, height=height_threshold, distance=distance_samples, prominence=0)
    
    # Compute median peak distance to set a minimum separation threshold
    if len(initial_peaks) > 1:
        median_distance = np.median(np.diff(initial_peaks))
    else:
        median_distance = distance_samples
    print(f"Initial Peak Count: {len(initial_peaks)}")
    print(f"New Distance: {median_distance * 0.7 / frequency:.2f} seconds")

    # Final peak detection with a distance threshold applied _after_ the height filter.
    final_peaks, _ = find_peaks(
        inverted_signal,
        height=height_threshold,
        distance=median_distance * 0.7,  # Use 70% of the median distance
        width=2  # A small width threshold to avoid noise spikes
    )

    print(f"Final Peak Count: {len(final_peaks)}")
    
    # -----------------------------------------------

    # Compute FFT and PSD if frequency domain analysis is enabled
    if frequency_domain:
        fft_freqs = np.fft.rfftfreq(len(force_mN), d=1/frequency)
        fft_magnitude = np.abs(np.fft.rfft(force_mN))
        psd_freqs, psd_values = welch(force_mN, fs=frequency, nperseg=len(force_mN)//8)

    # Determine number of subplots needed
    num_subplots = 1 + int(integrate) + int(frequency_domain) + int(derivative)
    fig, ax = plt.subplots(num_subplots, 1, figsize=(10, 2.5 * num_subplots))
    plt.subplots_adjust(hspace=0.6)
    if num_subplots == 1:
        ax = [ax]
    subplot_idx = 0

    # 1. Plot the original force signal (using force_corrected)
    ax[subplot_idx].plot(snippet['Time (seconds)'], force_mN, label=label)
    ax[subplot_idx].plot(snippet['Time (seconds)'].values[final_peaks], force_mN.values[final_peaks], "x", color='orange', label="New Cutting Cycle")
    ax[subplot_idx].set_title("Force Sensor Data")
    ax[subplot_idx].set_xlabel('Time (seconds)')
    ax[subplot_idx].set_ylabel('Force (mN)')
    ax[subplot_idx].legend()
    ax[subplot_idx].grid(True)
    subplot_idx += 1

    # 2. Plot the derivative if enabled
    if derivative:
        inverted_derivative = -derivative_signal
        if frequency_domain and cutoff_hz is not None:
            derivative_label = f"Filtered First Derivative (cutoff {cutoff_hz:.2f} Hz)"
        else:
            derivative_label = "First Derivative"
        ax[subplot_idx].plot(snippet['Time (seconds)'], inverted_derivative, 
                               label=derivative_label, 
                               color="purple")
        # Plot initial peaks (blue circles)
        ax[subplot_idx].plot(snippet['Time (seconds)'].values[initial_peaks], inverted_derivative[initial_peaks], "o", color='blue', label="Initial Peaks")
        # Plot final refined peaks (orange X's)
        ax[subplot_idx].plot(snippet['Time (seconds)'].values[final_peaks], 
                             inverted_derivative[final_peaks], 
                             "x", color='orange', label="Peaks")
        # Visual threshold marker for height threshold
        ax[subplot_idx].axhline(height_threshold, color='red', linestyle="dashed")
        # Annotate minimum distance (in seconds)
        min_distance_sec = (median_distance * 0.7) / frequency
        ax[subplot_idx].text(0.05, 0.9, f"Min Dist: {min_distance_sec:.2f} s", 
                             transform=ax[subplot_idx].transAxes, fontsize=10, color='black')
        ax[subplot_idx].set_title("First Derivative of Force Data with Peaks")
        ax[subplot_idx].set_xlabel('Time (seconds)')
        ax[subplot_idx].set_ylabel('- d(Force)/dt')
        ax[subplot_idx].legend()
        ax[subplot_idx].grid(True)
        subplot_idx += 1

    # 3. Plot frequency domain analysis if enabled
    if frequency_domain:
        ax[subplot_idx].plot(fft_freqs, fft_magnitude, label="FFT Magnitude", color="blue", alpha=0.7)
        ax[subplot_idx].axvline(cutoff_hz, color='red', linestyle='dashed', label=f"Cutoff Frequency ({cutoff_hz:.2f} Hz)")
        ax[subplot_idx].set_title("Frequency Domain Analysis (FFT)")
        ax[subplot_idx].set_xlabel('Frequency (Hz)')
        ax[subplot_idx].set_ylabel('Magnitude')
        ax[subplot_idx].legend()
        ax[subplot_idx].grid(True)
        subplot_idx += 1

    # 4. Plot the integration if selected
    if integrate:
        ax[subplot_idx].plot(snippet['Time (seconds)'], integrated_signal, label="Integrated Force", color="red")
        ax[subplot_idx].plot(snippet['Time (seconds)'].values[final_peaks], integrated_signal[final_peaks], "x", color='orange', label="Integrated Peaks")
        ax[subplot_idx].set_title("Integrated Force Sensor Data with Peaks")
        ax[subplot_idx].set_xlabel('Time (seconds)')
        ax[subplot_idx].set_ylabel('∫(Force - Mean) dt')
        ax[subplot_idx].legend()
        ax[subplot_idx].grid(True)

    plt.tight_layout()
    plt.show()

    return final_peaks