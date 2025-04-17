import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Script to plot LFP data from the NWB file

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get LFP data
lfp_data = nwb.processing["ecephys"].data_interfaces["LFP"].data
lfp_rate = nwb.processing["ecephys"].data_interfaces["LFP"].rate
electrodes_table = nwb.processing["ecephys"].data_interfaces["LFP"].electrodes.table

# Select a subset of electrodes (e.g., the first 4)
electrode_ids = electrodes_table.id[:4]  # Get the first 4 electrode IDs
selected_electrode_indices = np.where(np.isin(electrodes_table.id[:], electrode_ids))[0] # Find the indices of selected electrodes

# Select a time window (e.g., the first 1 second)
start_time = 0
end_time = 1
start_index = int(start_time * lfp_rate)
end_index = int(end_time * lfp_rate)

# Load the LFP data for the selected electrodes and time window
lfp_data_subset = lfp_data[start_index:end_index, selected_electrode_indices]

# Create a time axis
time_axis = np.linspace(start_time, end_time, end_index - start_index)

# Plot the LFP data
plt.figure(figsize=(10, 6))
for i, electrode_index in enumerate(selected_electrode_indices):
    plt.plot(time_axis, lfp_data_subset[:, i] + i*20 , label=f"Electrode {electrodes_table.id[electrode_index]}") # the i*20 is used to vertically space the signals

plt.xlabel("Time (s)")
plt.ylabel("LFP (uV)")
plt.title("LFP Data for Selected Electrodes")
plt.legend()
plt.savefig("explore/lfp_plot.png")

print("LFP plot saved to explore/lfp_plot.png")