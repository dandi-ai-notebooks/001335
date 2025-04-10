# This script loads a small LFP data snippet (first 1 second, ~2500 samples, 8 channels) and plots it to visualize filter quality and activity.

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

lfp_obj = nwb.processing["ecephys"].data_interfaces["LFP"]

# Sample rate
rate = lfp_obj.rate

segment_duration_sec = 1
num_samples = int(rate * segment_duration_sec)

# Select first 8 channels
num_channels = 8

# Extract small segment
lfp_data = lfp_obj.data[0:num_samples, 0:num_channels]

time = np.arange(num_samples) / rate

plt.figure(figsize=(12, 6))
for ch in range(num_channels):
    plt.plot(time, lfp_data[:, ch] * 1e3 + ch*2, label=f'Ch {ch}')  # scale to mV, offset

plt.xlabel('Time (s)')
plt.ylabel('Amplitude + offset (mV)')
plt.title('First 1 second of LFP data (8 channels)')
plt.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('tmp_scripts/lfp_snippet.png')
io.close()