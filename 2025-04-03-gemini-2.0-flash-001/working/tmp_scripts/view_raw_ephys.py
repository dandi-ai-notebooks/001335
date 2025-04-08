import pynwb
import numpy as np
import matplotlib.pyplot as plt
import lindi

# Script to visualize a snippet of raw electrophysiology data
try:
    # Load the NWB file
    f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

    # Select an electrode
    electrode_id = 0
    start_time = 10 # s
    duration = 2 # s

    # Get the LFP data
    raw_timeseries = nwb.processing['ecephys']['LFP'] # Changed this line!
    num_electrodes = len(raw_timeseries.electrodes.id)

    # Extract the samples for that electrode and plot them
    start_ind = int(start_time * raw_timeseries.rate)
    end_ind = int((start_time + duration) * raw_timeseries.rate)
    data = raw_timeseries.data[start_ind:end_ind, electrode_id] # Changed this line!
    times = np.linspace(start_time, start_time + duration, len(data)) # Changed this line

    plt.figure(figsize=(10, 4))
    plt.plot(times, data)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (uV)")
    plt.title(f"Raw Electrophysiology Data for Electrode {electrode_id}")

    plt.savefig("tmp_scripts/view_raw_ephys.png")
    plt.close()

except Exception as e:
    print(f"Error loading or processing NWB file: {e}")