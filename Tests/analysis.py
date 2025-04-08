def compare_experiments(csv_path_A, fragments_A, csv_path_B, fragments_B, output_csv_path='comparison_summary.csv', visualize=True):
    import csv
    import numpy as np
    from scipy.signal import welch

    def extract_sampling_rate(csv_path):
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        scan_rate_line = [line for line in lines if 'Scan Rate:' in line]
        if not scan_rate_line:
            raise ValueError(f"Could not find 'Scan Rate:' in metadata of {csv_path}.")
        return int(scan_rate_line[0].split(":")[1].strip().replace('"', ''))

    def extract_metrics(fragments, sampling_rate):
        avg_forces = []
        std_forces = []
        peak_forces = []
        durations = []
        chatter_indices = []
        onset_slopes = []

        for i, frag in enumerate(fragments):
            # Determine if frag is a NumPy array or a DataFrame
            if isinstance(frag, np.ndarray):
                force = frag
            else:
                if 'AI0 (V)' in frag.columns:
                    force = frag['AI0 (V)'].values
                else:
                    print(f"Skipping fragment {i}: missing 'AI0 (V)' column.")
                    continue

            print(f"Processing fragment {i} with length: {len(force)}")
            if len(force) == 0:
                print(f"Skipping fragment {i}: empty array.")
                continue

            # Compute metrics
            avg = np.mean(force)
            std = np.std(force)
            peak = np.max(force)
            duration = len(force) / sampling_rate

            freqs, psd = welch(force, fs=sampling_rate)
            chatter_band = psd[freqs > 15].sum()
            total_energy = psd.sum()
            chatter_idx = chatter_band / total_energy if total_energy > 0 else 0

            # Compute slope over 1 second window around the peak index
            local_peak_idx = np.argmax(force)
            start_idx = max(0, local_peak_idx - int(sampling_rate))
            end_idx = min(len(force), local_peak_idx + int(sampling_rate))
            x_vals = np.arange(end_idx - start_idx) / sampling_rate
            y_vals = force[start_idx:end_idx]
            slope = np.polyfit(x_vals, y_vals, 1)[0] if len(x_vals) > 1 else 0

            print(f"Fragment {i} metrics: avg={avg:.4f}, std={std:.4f}, peak={peak:.4f}, duration={duration:.4f}, chatter={chatter_idx:.4f}, slope={slope:.4f}")

            avg_forces.append(avg)
            std_forces.append(std)
            peak_forces.append(peak)
            durations.append(duration)
            chatter_indices.append(chatter_idx)
            onset_slopes.append(slope)

        return {
            'Average Force': avg_forces,
            'Force Std Dev': std_forces,
            'Peak Force': peak_forces,
            'Cut Duration': durations,
            'Chatter Index': chatter_indices,
            'Onset Slope': onset_slopes
        }

    rate_A = extract_sampling_rate(csv_path_A)
    rate_B = extract_sampling_rate(csv_path_B)

    metrics_A = extract_metrics(fragments_A, rate_A)
    metrics_B = extract_metrics(fragments_B, rate_B)

    summary = []
    for metric in metrics_A:
        data_A = metrics_A[metric]
        data_B = metrics_B[metric]
        mean_A, std_A = np.mean(data_A), np.std(data_A)
        mean_B, std_B = np.mean(data_B), np.std(data_B)

        value_A = f"{mean_A:.4f} ± {std_A:.4f}"
        value_B = f"{mean_B:.4f} ± {std_B:.4f}"

        # Compute relative % change unless mean_A is 0
        if mean_A != 0:
            diff = 100 * (mean_B - mean_A) / abs(mean_A)
            comparison = f"{diff:.2f}%"
        else:
            comparison = "N/A"

        summary.append([metric, value_A, value_B, comparison])

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Experiment A', 'Experiment B', 'Comparison'])
        writer.writerows(summary)

    print(f"Comparison analysis saved to {output_csv_path}")
    if visualize:
        import matplotlib.pyplot as plt
        metrics_names = list(metrics_A.keys())
        n_metrics = len(metrics_names)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics_names):
            data_A = np.array(metrics_A[metric])
            data_B = np.array(metrics_B[metric])
            # Create x-axis as fragment indices
            x_A = np.arange(len(data_A))
            x_B = np.arange(len(data_B))
            
            # Plot the metrics for each experiment using different markers/colors
            ax.plot(x_A, data_A, 'o-', label='Experiment A', color='blue')
            ax.plot(x_B, data_B, 'o-', label='Experiment B', color='red')
            ax.set_title(metric)
            ax.set_xlabel('Fragment Index')
            unit = 'Volts' if 'Force' in metric else 'Seconds' if 'Duration' in metric else 'Ratio' if 'Chatter' in metric else 'V/s'
            ax.set_ylabel(f"{metric} ({unit})")
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    return summary
