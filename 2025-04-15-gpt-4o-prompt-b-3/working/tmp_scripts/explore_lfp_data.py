"""
This script loads the NWB file from the specified URL, extracts a subset of the LFP data, and generates a line plot for a sample channel. The plot is saved as a PNG file in the tmp_scripts directory.
"""

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Extract LFP data from the first channel (0 to 500 samples)
lfp_data = nwb.processing["ecephys"].data_interfaces["LFP"].data[:500, 0]

# Generate the plot
plt.figure(figsize=(10, 5))
plt.plot(lfp_data, label='Channel 0')
plt.title('LFP Data from Channel 0')
plt.xlabel('Sample Number')
plt.ylabel('Voltage (V)')
plt.legend(loc='upper right')

# Save the plot as a PNG file
plt.savefig('tmp_scripts/lfp_channel0.png')
plt.close()

# Close the IO object
io.close()
remote_file.close()