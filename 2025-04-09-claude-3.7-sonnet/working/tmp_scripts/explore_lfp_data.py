"""
This script explores the LFP data in the NWB file, focusing on:
1. LFP signal visualization for selected channels
2. LFP heatmap across channels
3. Power spectral analysis
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

print("Loading LFP data...")
# Access LFP data
lfp = nwb.processing["ecephys"].data_interfaces["LFP"]
print(f"LFP data shape: {lfp.data.shape} (samples × channels)")
print(f"Sampling rate: {lfp.rate} Hz")
print(f"Total duration: {lfp.data.shape[0] / lfp.rate:.2f} seconds")

# Get information about odor presentation intervals
print("\nExtracting odor presentation intervals...")
odor_intervals = {}
for interval_name, interval in nwb.intervals.items():
    if "Odor" in interval_name and "ON" in interval_name:
        odor_intervals[interval_name] = []
        for i in range(len(interval.id)):
            odor_intervals[interval_name].append((interval.start_time[i], interval.stop_time[i]))

# Extract electrode information into a pandas DataFrame
electrode_data = {
    'id': [],
    'location': [],
    'group_name': [], 
    'depth': [],
    'hemisphere': []
}

print("\nExtracting electrode information...")
for i in range(len(nwb.electrodes.id)):
    electrode_data['id'].append(i)
    electrode_data['location'].append(nwb.electrodes['location'][i])
    electrode_data['group_name'].append(nwb.electrodes['group_name'][i])
    electrode_data['depth'].append(nwb.electrodes['depth'][i])
    electrode_data['hemisphere'].append(nwb.electrodes['hemisphere'][i])

electrodes_df = pd.DataFrame(electrode_data)

# Extract 5 seconds of data around an odor presentation (Odor A)
print("\nExtracting sample segment around an odor presentation...")
if 'Odor A ON' in odor_intervals and len(odor_intervals['Odor A ON']) > 0:
    # Take the first presentation of Odor A
    start_time, stop_time = odor_intervals['Odor A ON'][0] 
    
    # Window: 2 seconds before to 3 seconds after odor onset
    window_start = int((start_time - 2) * lfp.rate) 
    window_end = int((start_time + 3) * lfp.rate)
    
    # Make sure indices are within bounds
    window_start = max(0, window_start)
    window_end = min(lfp.data.shape[0], window_end)
    
    # Extract data
    print(f"Extracting data from {(window_start / lfp.rate):.2f}s to {(window_end / lfp.rate):.2f}s")
    lfp_segment = lfp.data[window_start:window_end, :]
    
    # Create time vector
    time_vector = np.arange(window_start, window_end) / lfp.rate
    
    # Plot LFP traces for a few channels from different shanks
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Select channels from different electrode groups (one from each probe/shank combination)
    selected_channels = []
    for probe in ['imec0', 'imec1']:
        for shank in range(2):  # Just take shank 0 and 1 from each probe
            mask = electrodes_df['group_name'] == f"{probe}.shank{shank}"
            if any(mask):
                # Take the middle electrode in the shank
                channel = electrodes_df[mask].iloc[len(electrodes_df[mask]) // 2]['id']
                selected_channels.append(int(channel))
    
    if len(selected_channels) < 4:  # If we couldn't get 4 channels, just take the first 4
        selected_channels = list(range(min(4, lfp_segment.shape[1])))
    
    # Plot each channel
    for i, channel in enumerate(selected_channels[:4]):
        group_name = electrodes_df.loc[electrodes_df['id'] == channel, 'group_name'].values[0]
        hemisphere = electrodes_df.loc[electrodes_df['id'] == channel, 'hemisphere'].values[0]
        depth = electrodes_df.loc[electrodes_df['id'] == channel, 'depth'].values[0]
        
        # Plot the LFP trace
        axes[i].plot(time_vector, lfp_segment[:, channel])
        axes[i].set_ylabel("Voltage (V)")
        axes[i].set_title(f"Channel {channel} ({group_name}, {hemisphere}, Depth: {depth}μm)")
        
        # Add vertical line at odor onset
        axes[i].axvline(x=start_time, color='r', linestyle='--', label="Odor Onset")
        
        # Add vertical line at odor offset
        axes[i].axvline(x=stop_time, color='b', linestyle='--', label="Odor Offset")
        
        # Only add legend to the first plot to avoid clutter
        if i == 0:
            axes[i].legend()
    
    # Set common xlabel
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("tmp_scripts/lfp_traces.png")
    plt.close()
    
    # Create a heatmap of LFP activity across all channels
    # Normalize the data for better visualization
    plt.figure(figsize=(12, 8))
    
    # Take a small subset of channels to make the heatmap more readable
    # Let's take 16 channels (mix of different probes/shanks)
    selected_channels = []
    for probe in ['imec0', 'imec1']:
        for shank in range(4):  # Take all 4 shanks
            mask = electrodes_df['group_name'] == f"{probe}.shank{shank}"
            if any(mask):
                # Take the middle electrode in the shank
                channel = electrodes_df[mask].iloc[len(electrodes_df[mask]) // 2]['id']
                selected_channels.append(int(channel))
    
    # Organize the channels by probe, shank, and hemisphere
    # This will help make the heatmap more interpretable
    selected_channels_df = electrodes_df[electrodes_df['id'].isin(selected_channels)].copy()
    selected_channels_df['probe'] = selected_channels_df['group_name'].apply(lambda x: x.split('.')[0])
    selected_channels_df['shank'] = selected_channels_df['group_name'].apply(lambda x: int(x.split('.')[-1][-1]))
    
    # Sort by hemisphere, then probe, then shank
    selected_channels_df = selected_channels_df.sort_values(['hemisphere', 'probe', 'shank']).reset_index(drop=True)
    selected_channels = selected_channels_df['id'].tolist()
    
    # Create a subset of the LFP data with just the selected channels
    lfp_subset = lfp_segment[:, selected_channels]
    
    # Normalize each channel to its min and max for better visualization
    normalized_lfp = np.zeros_like(lfp_subset)
    for i in range(lfp_subset.shape[1]):
        channel_data = lfp_subset[:, i]
        channel_min = np.min(channel_data)
        channel_max = np.max(channel_data)
        normalized_lfp[:, i] = (channel_data - channel_min) / (channel_max - channel_min)
    
    # Create the heatmap
    plt.imshow(normalized_lfp.T, aspect='auto', origin='lower', 
               extent=[time_vector[0], time_vector[-1], 0, lfp_subset.shape[1]])
    
    # Add colorbar
    plt.colorbar(label='Normalized Amplitude')
    
    # Add vertical lines for odor onset and offset
    plt.axvline(x=start_time, color='r', linestyle='--', label="Odor Onset")
    plt.axvline(x=stop_time, color='b', linestyle='--', label="Odor Offset")
    
    # Create custom y-tick labels with electrode info
    y_tick_labels = []
    for i, channel_id in enumerate(selected_channels):
        channel_info = selected_channels_df.iloc[i]
        y_tick_labels.append(f"{channel_info['hemisphere']}\n{channel_info['group_name']}")
    
    # Set y-ticks and labels
    plt.yticks(np.arange(len(selected_channels)) + 0.5, y_tick_labels, fontsize=8)
    
    # Set title and labels
    plt.title("LFP Heatmap During Odor Presentation")
    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    
    plt.tight_layout()
    plt.savefig("tmp_scripts/lfp_heatmap.png")
    plt.close()
    
    # Power spectral analysis
    print("\nPerforming power spectral analysis...")
    plt.figure(figsize=(12, 8))
    
    # Calculate power spectrum for a few channels
    # Select 4 channels (one from each hemisphere/probe combination)
    selected_channels = []
    for hemisphere in ['Left', 'Right']:
        for probe in ['imec0', 'imec1']:
            mask = (electrodes_df['hemisphere'] == hemisphere) & (electrodes_df['group_name'].str.contains(probe))
            if any(mask):
                # Take the middle electrode of the group
                channel = electrodes_df[mask].iloc[len(electrodes_df[mask]) // 2]['id']
                selected_channels.append(int(channel))
    
    if len(selected_channels) < 4:  # If we couldn't get enough channels, just take the first ones
        selected_channels = list(range(min(4, lfp_segment.shape[1])))
    
    for i, channel in enumerate(selected_channels):
        group_name = electrodes_df.loc[electrodes_df['id'] == channel, 'group_name'].values[0]
        hemisphere = electrodes_df.loc[electrodes_df['id'] == channel, 'hemisphere'].values[0]
        
        # Calculate the power spectrum
        f, pxx = signal.welch(lfp_segment[:, channel], fs=lfp.rate, nperseg=2048, scaling='spectrum')
        
        # Only plot up to 100 Hz (typical LFP range)
        mask = f <= 100
        
        # Plot the power spectrum
        plt.semilogy(f[mask], pxx[mask], label=f"Ch {channel} ({hemisphere}, {group_name})")
    
    # Add labels and legend
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V^2/Hz)")
    plt.title("Power Spectrum of LFP Signal")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.7)
    
    # Mark common neural frequency bands
    freq_bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (12, 30),
        "Gamma": (30, 100)
    }
    
    for band, (low, high) in freq_bands.items():
        plt.axvspan(low, high, color=f"C{list(freq_bands.keys()).index(band)}", alpha=0.2, label=f"{band} ({low}-{high} Hz)")
    
    plt.tight_layout()
    plt.savefig("tmp_scripts/lfp_power_spectrum.png")
    plt.close()
    
    # Time-frequency analysis for one channel
    print("\nPerforming time-frequency analysis...")
    
    # Select a channel from imec0.shank0
    channel_mask = electrodes_df['group_name'] == 'imec0.shank0'
    if any(channel_mask):
        channel = electrodes_df[channel_mask].iloc[0]['id']
        
        plt.figure(figsize=(12, 8))
        
        # Calculate the spectrogram
        f, t, Sxx = signal.spectrogram(lfp_segment[:, int(channel)], fs=lfp.rate, nperseg=512, noverlap=384, scaling='spectrum')
        
        # Only plot up to 100 Hz
        f_mask = f <= 100
        
        # Plot the spectrogram
        plt.pcolormesh(t + time_vector[0], f[f_mask], 10 * np.log10(Sxx[f_mask, :]), shading='gouraud', cmap='viridis')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        
        # Add vertical lines for odor onset and offset
        plt.axvline(x=start_time, color='r', linestyle='--', label="Odor Onset")
        plt.axvline(x=stop_time, color='b', linestyle='--', label="Odor Offset")
        
        # Add labels and legend
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        group_name = electrodes_df.loc[electrodes_df['id'] == channel, 'group_name'].values[0]
        hemisphere = electrodes_df.loc[electrodes_df['id'] == channel, 'hemisphere'].values[0]
        plt.title(f"Spectrogram of LFP Signal - Channel {channel} ({hemisphere}, {group_name})")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("tmp_scripts/lfp_spectrogram.png")
    else:
        print("Warning: Couldn't find a channel in imec0.shank0 for time-frequency analysis.")
        
else:
    print("Could not find 'Odor A ON' intervals or no intervals present")