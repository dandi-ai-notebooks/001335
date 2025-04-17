"""
This script explores the spike data (units) in the hippocampus recording.
It examines unit properties, spike timing, and neural responses to odor stimuli.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get basic unit information
units_df = nwb.units.to_dataframe()
print(f"Total number of units (neurons): {len(units_df)}")

print("\nElectrode group distribution:")
for group, count in units_df['electrode_group'].value_counts().items():
    if hasattr(group, 'name'):
        print(f"- {group.name}: {count} units")

# Get depth and hemisphere information
if 'depth' in units_df.columns:
    print("\nDepth distribution:")
    depth_bins = np.linspace(units_df['depth'].min(), units_df['depth'].max(), 10)
    depth_counts, _ = np.histogram(units_df['depth'].dropna(), bins=depth_bins)
    for i, count in enumerate(depth_counts):
        print(f"- {depth_bins[i]:.0f}-{depth_bins[i+1]:.0f} µm: {count} units")

if 'hemisphere' in units_df.columns:
    print("\nHemisphere distribution:")
    for hemisphere, count in units_df['hemisphere'].value_counts().items():
        print(f"- {hemisphere}: {count} units")

# Get spike times for each unit
# We'll use a subset of units to avoid loading too much data
num_units_to_analyze = 10
units_to_analyze = np.random.choice(len(units_df), num_units_to_analyze, replace=False)
units_to_analyze.sort()  # Sort to keep indices in ascending order

print(f"\nAnalyzing spike times for {num_units_to_analyze} randomly selected units:")
unit_spike_times = []
for unit_idx in units_to_analyze:
    # Get spike times for this unit
    spike_times = nwb.units['spike_times'][unit_idx]
    unit_spike_times.append(spike_times)
    # Print basic spike statistics
    print(f"Unit {unit_idx} (ID: {units_df.iloc[unit_idx]['global_id']}):")
    print(f"- Number of spikes: {len(spike_times)}")
    if len(spike_times) > 0:
        print(f"- Time range: {spike_times[0]:.2f}s to {spike_times[-1]:.2f}s")
        print(f"- Mean firing rate: {len(spike_times) / (spike_times[-1] - spike_times[0]):.2f} Hz")

# Get block intervals
block_intervals = {}
for block_name in ["Block 1", "Block 2", "Block 3"]:
    block_df = nwb.intervals[block_name].to_dataframe()
    block_intervals[block_name] = (block_df['start_time'].values[0], block_df['stop_time'].values[0])

# Get odor presentation intervals
odor_intervals = {}
for odor in ["A", "B", "C", "D", "E", "F"]:
    interval_name = f"Odor {odor} ON"
    if interval_name in nwb.intervals:
        odor_intervals[odor] = nwb.intervals[interval_name].to_dataframe()

# Create a plot of spike rasters around odor presentations
def plot_odor_triggered_raster(odor, time_window=(-1, 3)):
    """Plot spike rasters triggered by presentations of a specific odor"""
    if odor not in odor_intervals:
        print(f"No intervals found for Odor {odor}")
        return
    
    odor_df = odor_intervals[odor]
    
    # Select a subset of odor presentations to keep the plot readable
    max_presentations = 20
    if len(odor_df) > max_presentations:
        # Sample presentations from different parts of the experiment
        indices = np.linspace(0, len(odor_df)-1, max_presentations, dtype=int)
        odor_df_subset = odor_df.iloc[indices]
    else:
        odor_df_subset = odor_df
    
    plt.figure(figsize=(10, 8))
    
    for unit_num, (unit_idx, spike_times) in enumerate(zip(units_to_analyze, unit_spike_times)):
        for trial_num, (_, presentation) in enumerate(odor_df_subset.iterrows()):
            # Get odor onset time
            odor_onset = presentation['start_time']
            
            # Find spikes within the time window around odor onset
            window_start = odor_onset + time_window[0]
            window_end = odor_onset + time_window[1]
            
            # Filter spikes within this window
            trial_spikes = spike_times[(spike_times >= window_start) & (spike_times <= window_end)]
            
            # Plot raster for this trial
            if len(trial_spikes) > 0:
                trial_y = trial_num + unit_num * (len(odor_df_subset) + 1)  # Offset for each unit
                plt.scatter(trial_spikes - odor_onset, np.ones_like(trial_spikes) * trial_y, 
                           marker='|', s=20, color=f'C{unit_num%10}')
    
    # Add odor onset line
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Odor Onset')
    
    # Add odor offset line (assuming 2s duration based on average)
    plt.axvline(x=2, color='blue', linestyle='--', alpha=0.5, label='Odor Offset (approx)')
    
    plt.xlabel('Time relative to odor onset (s)')
    plt.ylabel('Trial × Unit')
    plt.title(f'Spike Raster Plot for Odor {odor} Presentations')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.savefig(f'explore/spike_raster_odor_{odor}.png')

# Create a combined plot showing firing rates for different odors
def plot_odor_firing_rate_comparison():
    """Plot average firing rates for different odors"""
    # Define the time window around odor onset
    time_window = (-1, 3)  # 1 second before to 3 seconds after odor onset
    bin_size = 0.1  # 100 ms bins
    
    # Create time bins
    bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    plt.figure(figsize=(12, 8))
    
    for unit_num, (unit_idx, spike_times) in enumerate(zip(units_to_analyze[:5], unit_spike_times[:5])):
        plt.subplot(5, 1, unit_num + 1)
        
        for odor in ["A", "B", "C", "D", "E", "F"]:
            if odor not in odor_intervals:
                continue
                
            odor_df = odor_intervals[odor]
            all_psth = []
            
            # Calculate PSTH for each odor presentation
            for _, presentation in odor_df.iterrows():
                odor_onset = presentation['start_time']
                window_start = odor_onset + time_window[0]
                window_end = odor_onset + time_window[1]
                
                # Filter spikes within this window
                trial_spikes = spike_times[(spike_times >= window_start) & (spike_times <= window_end)]
                
                # Convert to relative time
                relative_spikes = trial_spikes - odor_onset
                
                # Create histogram of spike counts
                hist, _ = np.histogram(relative_spikes, bins=bins)
                
                # Convert to firing rate (Hz)
                firing_rate = hist / bin_size
                all_psth.append(firing_rate)
            
            # Average across trials
            if all_psth:
                avg_psth = np.mean(all_psth, axis=0)
                sem_psth = np.std(all_psth, axis=0) / np.sqrt(len(all_psth))
                
                # Plot average firing rate with error bands
                plt.plot(bin_centers, avg_psth, label=f'Odor {odor}')
                plt.fill_between(bin_centers, avg_psth - sem_psth, avg_psth + sem_psth, alpha=0.2)
        
        # Add odor presentation time
        plt.axvspan(0, 2, color='gray', alpha=0.2, label='Odor ON')
        
        plt.title(f'Unit {unit_idx} (ID: {units_df.iloc[unit_idx]["global_id"]})')
        if unit_num == 0:
            plt.legend(loc='upper right')
        if unit_num == 4:  # Only add x-label on bottom subplot
            plt.xlabel('Time relative to odor onset (s)')
        plt.ylabel('Firing Rate (Hz)')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('explore/odor_firing_rate_comparison.png')

# Create raster plots for each odor
for odor in ["A", "B", "C"]:  # Just do ABC to avoid too many plots
    plot_odor_triggered_raster(odor)

# Create firing rate comparison plot
plot_odor_firing_rate_comparison()

print("\nExploration complete! Check the generated plots.")