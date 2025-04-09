# This script explores the NWB file for Dandiset 001335.
# It loads basic metadata and plots short snippets of LFP data from several channels.
# Plot is saved to PNG to avoid blocking. Prints key info to console.

import remfile
import h5py
import pynwb
import matplotlib.pyplot as plt
import numpy as np

url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"

print("Opening NWB file...")
file = remfile.File(url)
f = h5py.File(file, 'r')
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Subject ID: {nwb.subject.subject_id}, Species: {nwb.subject.species}, Sex: {nwb.subject.sex}")

lfp = nwb.processing["ecephys"].data_interfaces["LFP"]

print(f"LFP sampling rate: {lfp.rate} Hz")
print(f"LFP data shape: {lfp.data.shape}")

# Load a short segment: first 5000 samples (~2 seconds)
num_samples = 5000
num_channels = min(4, lfp.data.shape[1])

try:
    segment = lfp.data[:num_samples, :num_channels]
except Exception as e:
    print(f"Error loading LFP data segment: {e}")
    segment = None

if segment is not None:
    times = np.arange(num_samples) / lfp.rate
    plt.figure(figsize=(12, 6))
    for i in range(num_channels):
        plt.plot(times, segment[:, i] * 1e6 + i * 300, label=f'Ch {i}')  # scaled to uV, offset channels
    plt.xlabel("Time (s)")
    plt.ylabel("LFP (uV, offset per channel)")
    plt.title("LFP signals (first ~2s) from first few channels")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tmp_scripts/sample_lfp.png")
    plt.close()
    print("LFP plot saved to tmp_scripts/sample_lfp.png")
else:
    print("Skipping plot due to data load error.")

io.close()