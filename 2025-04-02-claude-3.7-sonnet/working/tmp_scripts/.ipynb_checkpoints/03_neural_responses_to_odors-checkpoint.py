"""
This script analyzes neural responses to odor presentations by:
1. Aligning spike times to odor onset events
2. Creating raster plots and PSTHs for selected units
3. Comparing responses to different odors
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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

# Get odor presentation times
odors = {}
for odor in ["A", "B", "C", "D", "E", "F"]:
    odor_intervals = nwb.intervals[f"Odor {odor} ON"]
    odors[odor] = {
        'starts': odor_intervals["start_time"][:],
        'stops': odor_intervals["stop_time"][:]
    }

# Get units and spike times
print("\n===== Analyzing Neural Responses to Odors =====")
units = nwb.units
spike_counts = []
for i in range(len(units['id'].data)):
    spike_times = units["spike_times"][i]
    spike_counts.append(len(spike_times))
spike_counts = np.array(spike_counts)

# Select units with high, medium and low spike counts for analysis
units_to_analyze = []
units_to_analyze.append(np.argmax(spike_counts))  # Highest firing unit
units_to_analyze.append(np.argsort(spike_counts)[len(spike_counts)//2])  # Median firing unit
units_to_analyze.append(np.argsort(spike_counts)[len(spike_counts)//4])  # Lower quartile firing unit

print(f"Selected units for analysis: {units_to_analyze}")
for i, unit_idx in enumerate(units_to_analyze):
    print(f"Unit {units['id'].data[unit_idx]}: {spike_counts[unit_idx]} spikes")

# Function to create raster plot and PSTH for a unit's response to odor onset
def plot_unit_odor_response(unit_idx, odor_label, window=(-1, 3), bin_size=0.05):
    unit_id = units['id'].data[unit_idx]
    spike_times = units["spike_times"][unit_idx]
    
    # Get list of odor onset times
    onset_times = odors[odor_label]['starts']
    
    # For each odor onset, get spikes within window
    aligned_spikes = []
    for onset in onset_times:
        # Find spikes within window relative to onset
        mask = (spike_times >= onset + window[0]) & (spike_times <= onset + window[1])
        relative_times = spike_times[mask] - onset
        aligned_spikes.append(relative_times)
    
    # Create figure with two subplots (raster and PSTH)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Create raster plot
    for i, spikes in enumerate(aligned_spikes):
        ax1.plot(spikes, np.ones_like(spikes) * i, '|', color='black', markersize=4)
    
    ax1.set_ylabel('Trial #')
    ax1.set_title(f'Unit {unit_id} Response to Odor {odor_label}')
    
    # Create PSTH
    bins = np.arange(window[0], window[1], bin_size)
    all_spikes = np.concatenate(aligned_spikes)
    hist, bin_edges = np.histogram(all_spikes, bins=bins)
    # Convert to firing rate (Hz)
    firing_rate = hist / (len(onset_times) * bin_size)
    
    ax2.bar(bin_edges[:-1], firing_rate, width=bin_size, color='blue', alpha=0.6)
    ax2.axvline(x=0, color='red', linestyle='--', label='Odor Onset')
    ax2.axvline(x=2, color='red', linestyle=':', label='Approx. Odor Offset')
    ax2.set_xlabel('Time from Odor Onset (s)')
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'tmp_scripts/unit_{unit_id}_odor_{odor_label}_response.png', dpi=150)
    plt.close()
    
    return firing_rate, bins

# Analyze responses to each odor for the highest firing unit
highest_unit_idx = units_to_analyze[0]
highest_unit_id = units['id'].data[highest_unit_idx]

# Get blocks where each odor is presented
block1_odors = ["A", "B", "C"]  # Based on our exploration of Block 1
block2_odors = ["D", "E", "F"]  # Based on our exploration of Block 2

# Plot odor responses for highest firing unit
print(f"\nPlotting odor responses for unit {highest_unit_id}")

# Store firing rates for comparison
firing_rates = {}

# Analyze Block 1 odors
for odor in block1_odors:
    print(f"Analyzing response to Odor {odor}...")
    fr, bins = plot_unit_odor_response(highest_unit_idx, odor)
    firing_rates[odor] = fr

# Analyze Block 2 odors
for odor in block2_odors:
    print(f"Analyzing response to Odor {odor}...")
    fr, bins = plot_unit_odor_response(highest_unit_idx, odor)
    firing_rates[odor] = fr

# Compare response profiles across odors within Block 1
plt.figure(figsize=(10, 6))
bin_centers = (bins[:-1] + bins[1:]) / 2

for odor in block1_odors:
    plt.plot(bin_centers, firing_rates[odor], label=f'Odor {odor}')

plt.axvline(x=0, color='red', linestyle='--', label='Odor Onset')
plt.axvline(x=2, color='red', linestyle=':', label='Approx. Odor Offset')
plt.xlabel('Time from Odor Onset (s)')
plt.ylabel('Firing Rate (Hz)')
plt.title(f'Unit {highest_unit_id} Responses to Block 1 Odors')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'tmp_scripts/unit_{highest_unit_id}_block1_odor_comparison.png', dpi=150)
plt.close()

# Compare response profiles across odors within Block 2
plt.figure(figsize=(10, 6))

for odor in block2_odors:
    plt.plot(bin_centers, firing_rates[odor], label=f'Odor {odor}')

plt.axvline(x=0, color='red', linestyle='--', label='Odor Onset')
plt.axvline(x=2, color='red', linestyle=':', label='Approx. Odor Offset')
plt.xlabel('Time from Odor Onset (s)')
plt.ylabel('Firing Rate (Hz)')
plt.title(f'Unit {highest_unit_id} Responses to Block 2 Odors')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'tmp_scripts/unit_{highest_unit_id}_block2_odor_comparison.png', dpi=150)
plt.close()

print("Script completed successfully.")