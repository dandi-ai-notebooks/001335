# This script visualizes spike times of the first 10 units over the first 5 seconds as a raster plot, to assess the quality and pattern of spiking data.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"

file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwb = io.read()

units = nwb.units
unit_ids = list(units.id[:])

max_units = min(10, len(unit_ids))
time_window = 5  # seconds

plt.figure(figsize=(10, 6))

for idx in range(max_units):
    unit_id = unit_ids[idx]
    spike_times = units['spike_times'][idx]
    # Plot spikes within first few seconds
    mask = (spike_times >= 0) & (spike_times <= time_window)
    plt.vlines(spike_times[mask], idx + 0.5, idx + 1.5)

plt.xlabel('Time (s)')
plt.ylabel('Unit ID')
plt.yticks(np.arange(1, max_units + 1), unit_ids[:max_units])
plt.title(f'Spike raster of first {max_units} units (first {time_window} seconds)')

plt.tight_layout()
plt.savefig('tmp_scripts/spike_raster.png')
io.close()