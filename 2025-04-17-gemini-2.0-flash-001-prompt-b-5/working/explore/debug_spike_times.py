import pynwb
import h5py
import remfile
import numpy as np

# Script to debug the spike_times data

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get spike times from the units table
units_table = nwb.units
unit_ids = units_table.id[:]

print(f"Unit IDs: {unit_ids}")

# Access a unit's spike times
spike_times = units_table.spike_times[unit_ids[0]]

print(f"Type of spike_times: {type(spike_times)}")
print(f"spike_times: {spike_times}")