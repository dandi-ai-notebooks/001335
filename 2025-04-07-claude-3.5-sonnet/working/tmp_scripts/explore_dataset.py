"""
This script explores the basic structure of the dataset including:
- Session information
- Odor presentation blocks timing
- Basic electrode and unit statistics
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get block timings
blocks = ['Block 1', 'Block 2', 'Block 3']
block_starts = []
block_durations = []

for block in blocks:
    interval = nwb.intervals[block]
    start = interval['start_time'][0]
    stop = interval['stop_time'][0]
    block_starts.append(start)
    block_durations.append(stop - start)

# Plot block durations
plt.figure(figsize=(10, 5))
plt.bar(blocks, block_durations)
plt.title('Duration of Each Experimental Block')
plt.ylabel('Duration (seconds)')
plt.savefig('tmp_scripts/block_durations.png')
plt.close()

# Get electrode depths and plot distribution
electrode_depths = nwb.electrodes['depth'].data[:]
plt.figure(figsize=(8, 6))
plt.hist(electrode_depths, bins=15)
plt.title('Distribution of Electrode Depths')
plt.xlabel('Depth (μm)')
plt.ylabel('Count')
plt.savefig('tmp_scripts/electrode_depths.png')
plt.close()

# Print key information
print("\nSession Information:")
print(f"Session Description: {nwb.session_description}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")
print(f"Species: {nwb.subject.species}")

print("\nRecording Information:")
print(f"Number of electrodes: {len(nwb.electrodes['id'].data[:])}")
print(f"Number of units: {len(nwb.units['id'].data)}")
print(f"LFP sampling rate: {nwb.processing['ecephys']['LFP'].rate} Hz")

print("\nBlock Timing Information:")
for block, start, duration in zip(blocks, block_starts, block_durations):
    print(f"{block}: starts at {start:.2f}s, duration {duration:.2f}s")