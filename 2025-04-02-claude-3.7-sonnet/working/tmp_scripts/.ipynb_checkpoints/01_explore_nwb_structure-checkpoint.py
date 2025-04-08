"""
This script explores the basic structure of the NWB file including:
1. Session information
2. Subject information
3. Block and odor intervals
4. Basic information about LFP data
5. Basic information about units (spike times)
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic session information
print("\n===== Session Information =====")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experiment description: {nwb.experiment_description}")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}")
print(f"Keywords: {nwb.keywords}")

# Print subject information
print("\n===== Subject Information =====")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")
print(f"Description: {nwb.subject.description}")

# Explore blocks
print("\n===== Blocks =====")
for block_name in ["Block 1", "Block 2", "Block 3"]:
    block = nwb.intervals[block_name]
    start_time = block["start_time"][0]
    stop_time = block["stop_time"][0]
    duration = stop_time - start_time
    print(f"{block_name}: Start={start_time:.2f}s, Stop={stop_time:.2f}s, Duration={duration:.2f}s")

# Explore odor presentations
print("\n===== Odor Presentations =====")
for odor in ["A", "B", "C", "D", "E", "F"]:
    odor_intervals = nwb.intervals[f"Odor {odor} ON"]
    num_presentations = len(odor_intervals["start_time"])
    avg_duration = np.mean(odor_intervals["stop_time"][:] - odor_intervals["start_time"][:])
    print(f"Odor {odor}: {num_presentations} presentations, Avg duration: {avg_duration:.4f}s")

# Plot odor presentation times for the first 300 seconds
print("\n===== Creating Odor Presentation Plot =====")
plt.figure(figsize=(12, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y']
odors = ['A', 'B', 'C', 'D', 'E', 'F']

for i, odor in enumerate(odors):
    odor_intervals = nwb.intervals[f"Odor {odor} ON"]
    starts = odor_intervals["start_time"][:]
    stops = odor_intervals["stop_time"][:]
    
    # Filter to first 300 seconds for visualization
    mask = starts < 300
    starts = starts[mask]
    stops = stops[mask]
    
    for j in range(len(starts)):
        plt.plot([starts[j], stops[j]], [i+1, i+1], color=colors[i], linewidth=4)

plt.yticks(range(1, len(odors)+1), [f'Odor {odor}' for odor in odors])
plt.xlabel('Time (s)')
plt.title('Odor Presentation Timeline (First 300s)')
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/odor_presentation_timeline.png', dpi=150, bbox_inches='tight')
plt.close()

# Explore LFP data
print("\n===== LFP Data Information =====")
lfp = nwb.processing["ecephys"]["LFP"]
print(f"LFP data shape: {lfp.data.shape}")
print(f"LFP sampling rate: {lfp.rate} Hz")
print(f"LFP duration: {lfp.data.shape[0]/lfp.rate:.2f} seconds")
print(f"Number of channels: {lfp.data.shape[1]}")

# Get electrode information
electrodes = nwb.electrodes
print("\n===== Electrode Information =====")
print(f"Number of electrodes: {len(electrodes['id'].data[:])}")
print(f"Electrode columns: {electrodes.colnames}")
locations = np.unique(electrodes['location'].data[:])
print(f"Electrode locations: {locations}")

# Explore units (neurons)
print("\n===== Units Information =====")
units = nwb.units
print(f"Number of units: {len(units['id'].data)}")
print(f"Units columns: {units.colnames}")

# Count spikes per unit
total_spikes = 0
spike_counts = []
for i in range(len(units['id'].data)):
    spike_times = units["spike_times"][i]
    spike_count = len(spike_times)
    spike_counts.append(spike_count)
    total_spikes += spike_count

print(f"Total number of spikes across all units: {total_spikes}")
print(f"Average spikes per unit: {np.mean(spike_counts):.2f}")
print(f"Min spikes in a unit: {np.min(spike_counts)}")
print(f"Max spikes in a unit: {np.max(spike_counts)}")

# Plot histogram of spike counts
plt.figure(figsize=(10, 6))
plt.hist(spike_counts, bins=30)
plt.xlabel('Number of Spikes per Unit')
plt.ylabel('Count')
plt.title('Distribution of Spike Counts Across Units')
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/spike_count_distribution.png', dpi=150)
plt.close()

print("Script completed successfully.")