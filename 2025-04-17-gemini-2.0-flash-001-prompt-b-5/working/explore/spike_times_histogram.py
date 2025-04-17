import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Script to plot a histogram of spike times from the NWB file

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get spike times from the units table
units_table = nwb.units
unit_ids = units_table.id[:]

# Collect spike times for each unit
all_spike_times = []
for i, unit_id in enumerate(unit_ids):
    spike_times = nwb.units['spike_times'][i]
    if isinstance(spike_times, np.ndarray) and len(spike_times) > 0: # Check if spike_times is an array and not empty
      all_spike_times.extend(spike_times)

# Convert to numpy array
all_spike_times = np.array(all_spike_times)

# Plot a histogram of the spike times
plt.figure(figsize=(10, 6))
plt.hist(all_spike_times, bins=100)
plt.xlabel("Time (s)")
plt.ylabel("Number of Spikes")
plt.title("Histogram of Spike Times")
plt.savefig("explore/spike_times_histogram.png")

print("Spike times histogram saved to explore/spike_times_histogram.png")