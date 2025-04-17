"""
This script explores the electrode information and LFP data in the hippocampus recording.
It examines electrode locations, groups, and visualizes LFP signals.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode information
electrode_df = nwb.electrodes.to_dataframe()
print(f"Total number of electrodes: {len(electrode_df)}")
print("\nElectrode groups:")
for group_name, count in electrode_df['group_name'].value_counts().items():
    print(f"- {group_name}: {count} electrodes")

print("\nElectrode locations:")
if 'location' in electrode_df.columns:
    for location, count in electrode_df['location'].value_counts().items():
        print(f"- {location}: {count} electrodes")
else:
    print("No location information available")

# Print the first few rows of the electrode dataframe
print("\nSample electrode information:")
print(electrode_df.head())

# Create a plot showing electrode depth by group
plt.figure(figsize=(10, 6))
if 'depth' in electrode_df.columns:
    for group_name in electrode_df['group_name'].unique():
        group_data = electrode_df[electrode_df['group_name'] == group_name]
        plt.scatter(group_data.index, group_data['depth'], 
                   label=group_name, alpha=0.7)
    plt.xlabel('Electrode Index')
    plt.ylabel('Depth (µm)')
    plt.title('Electrode Depth by Group')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('explore/electrode_depth.png')
else:
    print("Depth information not available for electrodes")

# Get LFP data
lfp = nwb.processing['ecephys'].data_interfaces['LFP']
print("\nLFP Information:")
print(f"- Number of channels: {lfp.data.shape[1]}")
print(f"- Number of timepoints: {lfp.data.shape[0]}")
print(f"- Sampling rate: {lfp.rate} Hz")
print(f"- Duration: {lfp.data.shape[0]/lfp.rate:.2f} seconds")
print(f"- Description: {lfp.description}")

# Extract a small chunk of LFP data (10 seconds from the beginning of Block 1)
# First determine the index for Block 1 start time
block1_start = nwb.intervals["Block 1"].to_dataframe()['start_time'].values[0]
start_idx = int(block1_start * lfp.rate)  # Convert time to samples
chunk_size = int(10 * lfp.rate)  # 10 seconds of data
lfp_chunk = lfp.data[start_idx:start_idx+chunk_size, :]

# Create time array for the extracted chunk
time_chunk = np.arange(chunk_size) / lfp.rate + block1_start

# Plot LFP data from 8 random channels
plt.figure(figsize=(15, 10))
random_channels = np.random.choice(lfp_chunk.shape[1], 8, replace=False)
for i, channel in enumerate(random_channels):
    # Normalize and offset each channel for visibility
    signal = lfp_chunk[:, channel]
    normalized_signal = (signal - signal.mean()) / signal.std()
    plt.plot(time_chunk, normalized_signal + i*5, 
             label=f"Channel {electrode_df.iloc[channel]['label']}")

plt.xlabel('Time (s)')
plt.ylabel('Normalized LFP')
plt.title('LFP Signals from 8 Random Channels')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig('explore/lfp_signals.png')

# Create a time-frequency analysis (spectrogram) for a single channel
from scipy import signal as sg

# Select one channel for the spectrogram
selected_channel = random_channels[0]
lfp_signal = lfp_chunk[:, selected_channel]

# Calculate spectrogram
f, t, Sxx = sg.spectrogram(lfp_signal, 
                          fs=lfp.rate, 
                          window='hann', 
                          nperseg=int(lfp.rate * 0.5),  # 0.5 second window
                          noverlap=int(lfp.rate * 0.3),  # 60% overlap
                          scaling='spectrum')

# Only show frequencies up to 100 Hz
f_mask = f <= 100
f_masked = f[f_mask]
Sxx_masked = Sxx[f_mask, :]

plt.figure(figsize=(15, 6))
plt.pcolormesh(t + block1_start, f_masked, 10 * np.log10(Sxx_masked), 
              shading='gouraud', cmap='viridis')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title(f'Spectrogram of LFP Channel {electrode_df.iloc[selected_channel]["label"]}')
plt.savefig('explore/lfp_spectrogram.png')

# Calculate average power spectrum across the whole 10-second chunk
plt.figure(figsize=(10, 6))
# Compute power spectral density using Welch's method
f, Pxx = sg.welch(lfp_signal, fs=lfp.rate, 
                 window='hann', 
                 nperseg=int(lfp.rate * 1.0),  # 1 second window
                 scaling='spectrum')
# Plot only up to 100 Hz
f_mask = f <= 100
plt.semilogy(f[f_mask], Pxx[f_mask])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V²/Hz)')
plt.title(f'Power Spectrum of LFP Channel {electrode_df.iloc[selected_channel]["label"]}')
plt.grid(True)
plt.savefig('explore/lfp_power_spectrum.png')

print("\nExploration complete! Check the generated plots.")