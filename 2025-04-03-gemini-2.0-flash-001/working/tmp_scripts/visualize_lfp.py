import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np

# Script to visualize LFP data from the NWB file
try:
    # Load the NWB file
    f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

    # Get LFP data
    lfp = nwb.processing["ecephys"]["LFP"]
    lfp_data = lfp.data
    electrodes = lfp.electrodes

    # Select a subset of electrodes and time points
    num_electrodes = 5
    num_timepoints = 1000
    electrode_ids = electrodes["id"].data[:num_electrodes]
    start_time = 0
    end_time = num_timepoints / lfp.rate

    #Get the data (must read into memory)
    data = lfp_data[:num_timepoints, :num_electrodes]

    # Create a time vector
    time = np.linspace(start_time, end_time, num_timepoints)

    # Plot the LFP data for each electrode
    plt.figure(figsize=(10, 6))
    for i in range(num_electrodes):
        plt.plot(time, data[:, i], label=f"Electrode {electrode_ids[i]}")

    plt.xlabel("Time (s)")
    plt.ylabel("LFP (mV)")
    plt.title("LFP Data for Selected Electrodes")
    plt.legend()
    plt.savefig("tmp_scripts/lfp_visualization.png")  # Save the plot to a file
    plt.close()

except Exception as e:
    print(f"Error loading or processing NWB file: {e}")