import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for interactive plots
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, welch
from pathlib import Path

def plot_force_data(filepath, skiprows=6, frequency=1000, ignore_AI0=False, ignore_AI1=False):
    # Load the data, skipping the first 6 rows and using the 7th row as header
    data = pd.read_csv(filepath, skiprows=skiprows)

    # Convert 'Sample' to seconds for the x-axis
    data['Sample'] = data['Sample'] / frequency

    # Plotting the data interactively
    plt.figure(figsize=(10, 5))
    if not ignore_AI0:
        plt.plot(data['Sample'], data['AI0 (V)'], label='AI0 (V)', marker=',', linestyle=None)
    if not ignore_AI1:
        plt.plot(data['Sample'], data['AI1 (V)'], label='AI1 (V)', marker=',', linestyle=None)
    plt.title('Interactive Force Sensor Data Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Force (Volts)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return data

def split_and_plot(csv_file, x_inputs, frequency=1000, ignore_AI0=False, ignore_AI1=False):
    # Load the CSV file
    data = pd.read_csv(csv_file)
    
    # Ensure x column exists (assuming the first column is x)
    if data.empty or len(data.columns) < 2:
        raise ValueError("CSV file must contain at least two columns (x and y data).")
    
    x_column = 'Sample'  # Use 'Sample' as the x column
    if not ignore_AI0 and 'AI0 (V)' in data.columns:
        ai0 = 'AI0 (V)'  # Use 'AI0 (V)'
    if not ignore_AI1 and 'AI1 (V)' in data.columns:
        ai1 = 'AI1 (V)'  # Use 'AI1 (V)'

    # Identify indices where to split the data
    split_indices = [data[data[x_column] >= x].index.min() for x in x_inputs]
    split_indices = [idx for idx in split_indices if not pd.isna(idx)]
    
    # Split the dataframe into snippets
    split_points = [0] + split_indices + [len(data)]
    snippets = [data.iloc[split_points[i]:split_points[i+1]] for i in range(len(split_points)-1)]
    
    # Plot the snippets
    fig, axes = plt.subplots(len(snippets), 1, figsize=(8, 4 * len(snippets)))
    plt.subplots_adjust(hspace=1.5)
    if len(snippets) == 1:
        axes = [axes]  # Ensure axes is iterable if only one plot
    
    for i, snippet in enumerate(snippets):
        if not ignore_AI0:
            axes[i].plot(snippet[x_column], snippet[ai0])
        if not ignore_AI1:
            axes[i].plot(snippet[x_column], snippet[ai1])
        axes[i].set_title(f'Snippet {i+1}')
        axes[i].set_xlabel('Time (seconds)')
        axes[i].set_ylabel('Force (Volts)')
        axes[i].grid()
    
    plt.tight_layout()
    plt.show()
    
    return snippets

def plot_peaks(snippet, frequency=1000, height=0.1, distance=30, prominence=None, width=None, label='Force Data'): # distance should be inputted in seconds
    
    if 'Time (seconds)' not in snippet.columns:
        snippet['Time (seconds)'] = snippet['Sample'] / frequency
    distance_samples = frequency * distance  # Convert distance from seconds to number of samples
    if 'Sample' not in snippet.columns or 'AI0 (V)' not in snippet.columns:
        raise ValueError("Snippet must contain 'Sample' and 'AI0 (V)' columns.")
    
    x_column = 'Time (seconds)'
    y_column = 'AI0 (V)'  # Select the correct column
    # Identify peaks
    std_force = np.std(snippet[y_column])
    peaks, _ = find_peaks(snippet[y_column], height=height, distance=distance_samples, prominence=2 * std_force, width=width)

    # Plot the snippet
    plt.figure(figsize=(10, 5))
    plt.plot(snippet[x_column], snippet[y_column], label=label)
    plt.plot(snippet[x_column].iloc[peaks], snippet[y_column].iloc[peaks], "x", label='Peaks')
    plt.title('Force Sensor Data with Peaks')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Force (Volts)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return peaks

def pad_fragment(fragment, target_length):
    """Pads fragment to ensure it matches target_length with the last value in the array."""
    fragment = np.asarray(fragment)  # Ensure fragment is a NumPy array
    current_length = len(fragment)
    
    if current_length == 0:
        return np.zeros(target_length)  # Return all-zero array if empty
    
    if current_length < target_length:
        return np.pad(fragment, (0, target_length - current_length), mode='constant', constant_values=fragment[-1])
        
    return fragment[:target_length]  # Trim excess if needed

def plot_peaks_overlay(snippets, peaks_list=None, labels=None, peak_output=None):
    if peak_output is not None:
        peaks_list = peak_output  # Use output from plot_sharp_peaks if provided

    num_snippets = len(snippets)
    if labels and len(labels) != num_snippets:
        raise ValueError("Number of labels must match the number of snippets.")
    
    fig, axes = plt.subplots(num_snippets, 1, figsize=(10, 5 * num_snippets))
    # Let tight_layout handle spacing automatically
    
    if num_snippets == 1:
        axes = [axes]  # Ensure axes is iterable if only one plot

    results = []
    
    # Determine global x and y limits
    global_x_min, global_x_max = float('inf'), float('-inf')
    global_y_min, global_y_max = float('inf'), float('-inf')

    for idx, (snippet, peaks, label) in enumerate(zip(snippets, peaks_list, labels)):
        if isinstance(snippet, (str, Path)):
            snippet = pd.read_csv(snippet, skiprows=6)

        x_column = 'Time (seconds)'
        # Convert "AI0 (V)" to "Force (mN)" if available
        if 'AI0 (V)' in snippet.columns:
            volts_per_lb = 2.2
            newtons_per_lb = 4.44822162
            conversion_factor = (newtons_per_lb / volts_per_lb) * 1000  # mN per Volt
            snippet['Force (mN)'] = (snippet['AI0 (V)'] - snippet['AI0 (V)'].mean()) * conversion_factor
            y_column = 'Force (mN)'
        else:
            y_column = snippet.columns[-1]

        if len(peaks) == 0:
            print(f"No valid peaks found in snippet {idx + 1}. Skipping.")
            continue

        # Extract fragments from 100 points before peak to 100 points before the next peak
        fragments = []
        num_peaks = len(peaks)

        for i, peak in enumerate(peaks):
            start = max(0, peak - 100)
            end = max(0, peaks[i + 1] - 100) if i < num_peaks - 1 else len(snippet)  # Ensure we don't exceed bounds
            fragments.append(snippet.iloc[start:end])

        # Ensure max_length is valid before proceeding
        if len(fragments) == 0:
            print(f"No valid fragments found for snippet {idx + 1}. Skipping.")
            continue

        max_length = max(len(fragment) for fragment in fragments) if fragments else 0
        x_range = np.linspace(-100, max_length - 100, max_length)  # Ensure consistency

        # Ensure x and y inputs have the same length
        interpolated_fragments = []
        for fragment in fragments:
            if len(fragment) > 0:
                # Create xp based on the fragment's original length
                xp = np.linspace(-100, len(fragment) - 100, len(fragment))
                # Use raw values (do not pad) so that xp and fp match in length
                fp = fragment[y_column].values
                # Interpolate fragment values to common x_range of length max_length
                interpolated_fragments.append(np.interp(x_range, xp, fp))
            else:
                interpolated_fragments.append(np.zeros(max_length))  # Handle empty fragments safely

        # Compute average fragment
        avg_fragment = np.mean(np.vstack(interpolated_fragments), axis=0)

        # Update global x and y limits
        global_x_min = min(global_x_min, x_range[0])
        global_x_max = max(global_x_max, x_range[-1])
        global_y_min = min(global_y_min, np.min(avg_fragment))
        global_y_max = max(global_y_max, np.max(avg_fragment))

        # Plot fragments without padding values
        colors = plt.cm.viridis(np.linspace(0, 1, len(interpolated_fragments)))  # Generate a color map

        for i, fragment in enumerate(interpolated_fragments):
            valid_indices = np.where(fragment != fragment[-1])[0]  # Exclude padding values
            axes[idx].plot(x_range[valid_indices], fragment[valid_indices], alpha=0.5, color=colors[i])

        # Add a simple label indicating the number of sections/peaks detected
        axes[idx].text(0.02, 0.95, f"Sections cut = {len(interpolated_fragments)}",
                       transform=axes[idx].transAxes, fontsize=10, color='black',
                       ha='left', va='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

        # Plot average fragment
        axes[idx].plot(x_range, avg_fragment, color='blue', linewidth=2, label=label)
        axes[idx].legend()

        # Set labels
        axes[idx].set_xlabel('Time (seconds)')
        axes[idx].set_ylabel("Force (mN)")
        axes[idx].grid()

        results.append([interpolated_fragments, avg_fragment])

    # Apply global x and y limits to all subplots
    for ax in axes:
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    return results

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

def simulate_real_time_force_plot(snippet, frequency=1000, duration=None):
    """
    Simulates real-time plotting of force data.
    snippet: either a path to a CSV file or a DataFrame with 'Time (seconds)' and 'AI0 (V)' columns.
    frequency: sampling rate in Hz.
    duration: maximum duration to simulate in seconds (optional).
    """
    # Load CSV if path is passed
    if isinstance(snippet, (str, Path)):
        df = pd.read_csv(snippet, skiprows=6)
        df['Sample'] = df['Sample'].astype(int)
        df['Time (seconds)'] = df['Sample'] / frequency
    else:
        df = snippet.copy()

    # Convert to mN
    volts_per_lb = 2.2
    newtons_per_lb = 4.44822162
    conversion_factor = (newtons_per_lb / volts_per_lb) * 1000  # mN per Volt
    force_mN = (df['AI0 (V)'] - df['AI0 (V)'].mean()) * conversion_factor

    times = df['Time (seconds)'].values
    interval = 1 / frequency

    if duration:
        max_idx = int(duration * frequency)
        times = times[:max_idx]
        force_mN = force_mN[:max_idx]

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, times[-1])
    ax.set_ylim(force_mN.min() * 1.1, force_mN.max() * 1.1)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Force (mN)")
    ax.set_title("Force Signal")

    x_data, y_data = [], []

    start_time = time.perf_counter()
    i = 0

    while i < len(times):
        elapsed = time.perf_counter() - start_time

        while i < len(times) and times[i] <= elapsed:
            x_data.append(times[i])
            y_data.append(force_mN.iloc[i])
            i += 1

        line.set_data(x_data, y_data)
        ax.set_xlim(max(0, elapsed - 5), elapsed + 0.5)
        plt.draw()
        plt.pause(0.01)

    plt.show()
