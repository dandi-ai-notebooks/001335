"""
This script explores:
1. Timing of odor presentations
2. LFP data from a subset of channels during odor presentations
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get all odor presentation intervals
odors = ['A', 'B', 'C', 'D', 'E', 'F']
odor_times = {}
for odor in odors:
    interval = nwb.intervals[f'Odor {odor} ON']
    odor_times[odor] = {
        'start': interval['start_time'][:],
        'stop': interval['stop_time'][:]
    }

# Plot odor presentation timing for first few trials
plt.figure(figsize=(15, 6))
for i, (odor, times) in enumerate(odor_times.items()):
    # Plot first 10 presentations
    for start, stop in zip(times['start'][:10], times['stop'][:10]):
        plt.plot([start, stop], [i, i], linewidth=4)
plt.yticks(range(len(odors)), odors)
plt.xlabel('Time (seconds)')
plt.title('First 10 Odor Presentations')
plt.grid(True)
plt.savefig('tmp_scripts/odor_timing.png')
plt.close()

# Get LFP data for a short window around first odor presentation
# Choose 3 channels spread across different depths
electrode_depths = nwb.electrodes['depth'].data[:]
depth_percentiles = [25, 50, 75]  # Use 25th, 50th, and 75th percentiles
channel_indices = []
for p in depth_percentiles:
    target_depth = np.percentile(electrode_depths, p)
    idx = np.argmin(np.abs(electrode_depths - target_depth))
    channel_indices.append(idx)
channel_indices = sorted(list(set(channel_indices)))  # Remove duplicates and sort

# Get LFP data
lfp = nwb.processing['ecephys']['LFP']
sampling_rate = lfp.rate

# Calculate time window: 1 second before to 2 seconds after first odor A presentation
start_time = odor_times['A']['start'][0]
window_start = int((start_time - 1) * sampling_rate)
window_end = int((start_time + 2) * sampling_rate)

# Get LFP data for selected channels in the time window
lfp_data = lfp.data[window_start:window_end, channel_indices]
time = np.arange(window_start, window_end) / sampling_rate

# Plot LFP traces
plt.figure(figsize=(12, 8))
for i, (data, idx) in enumerate(zip(lfp_data.T, channel_indices)):
    depth = electrode_depths[idx]
    plt.plot(time, data + i*200, label=f'Depth: {depth:.0f} μm')

plt.axvline(x=start_time, color='r', linestyle='--', label='Odor onset')
plt.axvline(x=odor_times['A']['stop'][0], color='r', linestyle=':', label='Odor offset')
plt.xlabel('Time (seconds)')
plt.ylabel('LFP (μV)')
plt.title('LFP at Different Depths During First Odor A Presentation')
plt.legend()
plt.grid(True)
plt.savefig('tmp_scripts/lfp_example.png')
plt.close()

# Print some statistics about odor presentations
print("\nOdor Presentation Statistics:")
for odor in odors:
    n_presentations = len(odor_times[odor]['start'])
    mean_duration = np.mean(odor_times[odor]['stop'] - odor_times[odor]['start'])
    print(f"\nOdor {odor}:")
    print(f"Number of presentations: {n_presentations}")
    print(f"Average duration: {mean_duration:.2f} seconds")