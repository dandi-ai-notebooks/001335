import pynwb
import lindi

# Script to explore the NWB file and print basic information
try:
    # Load the NWB file
    f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

    # Print basic information
    print(f"Session description: {nwb.session_description}")
    print(f"Identifier: {nwb.identifier}")
    print(f"Session start time: {nwb.session_start_time}")
    print(f"Experiment description: {nwb.experiment_description}")

    # List available intervals
    print("\\nAvailable intervals:")
    for interval_name in nwb.intervals:
        print(f"- {interval_name}")

    # List available processing modules
    print("\\nAvailable processing modules:")
    for module_name in nwb.processing:
        print(f"- {module_name}")

    print("\\nElectrode column names:")
    print(nwb.electrodes.colnames)
except Exception as e:
    print(f"Error loading or processing NWB file: {e}")