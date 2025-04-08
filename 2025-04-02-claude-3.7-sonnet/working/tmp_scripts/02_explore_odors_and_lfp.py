"""
This script creates improved visualizations and explores:
1. Odor presentation timing in each experimental block
2. LFP data samples
3. Electrode positions and depths
4. Basic time-frequency analysis of LFP
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import signal

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get block time intervals
blocks = {}
for block_name in ["Block 1", "Block 2", "Block 3"]:
    block = nwb.intervals[block_name]
    blocks[block_name] = {
        'start': block["start_time"][0],
        'stop': block["stop_time"][0]
    }

# Get odor presentation intervals
odors = {}
for odor in ["A", "B", "C", "D", "E", "F"]:
    odor_intervals = nwb.intervals[f"Odor {odor} ON"]
    odors[odor] = {
        'starts': odor_intervals["start_time"][:],
        'stops': odor_intervals["stop_time"][:]
    }

# Create improved odor presentation plots for each block
print("\n===== Creating Odor Presentation Plots =====")
colors = ['r', 'g', 'b', 'c', 'm', 'y']
odor_labels = ['A', 'B', 'C', 'D', 'E', 'F']

for block_name, block_times in blocks.items():
    block_start = block_times['start']
    block_stop = block_times['stop']
    
    plt.figure(figsize=(12, 6))
    
    # Plot the first 120 seconds of each block
    plot_duration = 120
    plot_end = min(block_start + plot_duration, block_stop)
    
    for i, odor in enumerate(odor_labels):
        starts = odors[odor]['starts']
        stops = odors[odor]['stops']
        
        # Filter to current block and plot duration
        mask = (starts >= block_start) & (starts < plot_end)
        block_starts = starts[mask]
        block_stops = stops[mask]
        
        # Plot each odor presentation interval
        for j in range(len(block_starts)):
            # Convert to block-relative time
            rel_start = block_starts[j] - block_start
            rel_stop = block_stops[j] - block_start
            plt.plot([rel_start, rel_stop], [i+1, i+1], color=colors[i], linewidth=4)
    
    plt.yticks(range(1, len(odor_labels)+1), [f'Odor {odor}' for odor in odor_labels])
    plt.xlabel('Time from block start (s)')
    plt.title(f'{block_name} Odor Presentation Timeline (First {plot_duration}s)')
    plt.xlim(0, plot_duration)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'tmp_scripts/{block_name.replace(" ", "_")}_odor_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()

# Explore electrode depths and positions
print("\n===== Exploring Electrode Information =====")
electrodes = nwb.electrodes
depths = electrodes['depth'].data[:]
hemisphere = electrodes['hemisphere'].data[:]

# Plot electrode depths
plt.figure(figsize=(8, 10))
plt.scatter(np.ones(len(depths)), depths, c=range(len(depths)), cmap='viridis', s=50)
plt.xlabel('Electrode Position')
plt.ylabel('Depth (µm)')
plt.title('Electrode Depths')
plt.colorbar(label='Electrode Number')
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/electrode_depths.png', dpi=150, bbox_inches='tight')
plt.close()

# Explore LFP data
print("\n===== Exploring LFP Data =====")
lfp = nwb.processing["ecephys"]["LFP"]

# Get a short segment of LFP data from each block for visualization
lfp_segments = {}
sampling_rate = lfp.rate
segment_duration = 1.0  # 1 second
samples_per_segment = int(segment_duration * sampling_rate)

for block_name, block_times in blocks.items():
    # Get LFP segment from middle of block
    block_middle = (block_times['start'] + block_times['stop']) / 2
    start_sample = int(block_middle * sampling_rate)
    
    # Limit to ensure we don't exceed array bounds
    if start_sample + samples_per_segment >= lfp.data.shape[0]:
        start_sample = lfp.data.shape[0] - samples_per_segment - 1
    
    # Get segment for all channels
    lfp_segments[block_name] = lfp.data[start_sample:start_sample+samples_per_segment, :]

# Plot LFP traces for first 8 channels in each block
for block_name, lfp_segment in lfp_segments.items():
    plt.figure(figsize=(12, 10))
    time = np.arange(lfp_segment.shape[0]) / sampling_rate
    
    # Plot first 8 channels
    for i in range(8):
        # Normalize and offset for display
        trace = lfp_segment[:, i] / np.max(np.abs(lfp_segment[:, i]))
        plt.plot(time, trace + i*2.2, label=f'Ch {i}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Channel')
    plt.title(f'{block_name} LFP Traces (Selected Channels)')
    plt.yticks(np.arange(0, 8*2.2, 2.2), [f'Ch {i}' for i in range(8)])
    plt.grid(True, alpha=0.3)
    plt.savefig(f'tmp_scripts/{block_name.replace(" ", "_")}_lfp_traces.png', dpi=150, bbox_inches='tight')
    plt.close()

# Calculate and plot spectrogram for one channel
print("\n===== Creating LFP Spectrogram =====")

# Choose block 1 and channel 0 for spectral analysis
lfp_segment = lfp_segments["Block 1"][:, 0]

# Calculate spectrogram
nperseg = int(0.2 * sampling_rate)  # 200 ms segments
noverlap = nperseg // 2
f, t, Sxx = signal.spectrogram(lfp_segment, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

# Plot spectrogram up to 100 Hz
freq_mask = f <= 100
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f[freq_mask], 10 * np.log10(Sxx[freq_mask, :]), shading='gouraud', cmap='viridis')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.title('LFP Spectrogram (Channel 0, Block 1)')
plt.savefig('tmp_scripts/lfp_spectrogram.png', dpi=150, bbox_inches='tight')
plt.close()

# Calculate and plot average waveform for a few units
print("\n===== Exploring Unit Waveforms =====")
units = nwb.units
waveforms = units['waveform_mean'].data[:]

# Select a few units with different spike counts
spike_counts = []
for i in range(len(units['id'].data)):
    spike_times = units["spike_times"][i]
    spike_counts.append(len(spike_times))
spike_counts = np.array(spike_counts)

# Choose units with low, medium, and high spike counts
low_idx = np.argmin(spike_counts)
high_idx = np.argmax(spike_counts)
med_idx = np.argsort(spike_counts)[len(spike_counts)//2]

plt.figure(figsize=(10, 6))
time_axis = np.arange(waveforms.shape[1]) / 30  # Assuming 30 kHz sampling

plt.plot(time_axis, waveforms[low_idx], label=f'Unit {units["id"].data[low_idx]}: {spike_counts[low_idx]} spikes')
plt.plot(time_axis, waveforms[med_idx], label=f'Unit {units["id"].data[med_idx]}: {spike_counts[med_idx]} spikes')
plt.plot(time_axis, waveforms[high_idx], label=f'Unit {units["id"].data[high_idx]}: {spike_counts[high_idx]} spikes')

plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (a.u.)')
plt.title('Average Spike Waveforms')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/spike_waveforms.png', dpi=150, bbox_inches='tight')
plt.close()

print("Script completed successfully.")