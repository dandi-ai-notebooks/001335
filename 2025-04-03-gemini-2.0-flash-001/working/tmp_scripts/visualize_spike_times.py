import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np

# Script to visualize spike times from the NWB file
try:
    # Load the NWB file
    f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

    # Get units data
    units = nwb.units
    spike_times = units["spike_times"]

    # Select a subset of units
    num_units = 5
    unit_ids = units["id"].data[:num_units]

    # Plot histograms of spike times for each unit
    plt.figure(figsize=(10, 6))
    for i in range(num_units):
        times = spike_times[i]
        plt.hist(times, bins=50, alpha=0.5, label=f"Unit {unit_ids[i]}")

    plt.xlabel("Time (s)")
    plt.ylabel("Number of Spikes")
    plt.title("Spike Time Histograms for Selected Units")
    plt.legend()
    plt.savefig("tmp_scripts/spike_time_histograms.png")  # Save the plot to a file
    plt.close()

except Exception as e:
    print(f"Error loading or processing NWB file: {e}")