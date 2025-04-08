# This script loads the NWB file and prints key metadata, intervals, LFP properties, electrodes, and unit counts.
# It is part of exploratory research for notebook generation.

import lindi
import pynwb

# Load NWB from optimized lindi JSON URL
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print("Session description:", nwb.session_description)
print("Identifier:", nwb.identifier)
print("Session start time:", nwb.session_start_time)

print("\nExperimenter(s):", nwb.experimenter)
print("Experiment description:", nwb.experiment_description)
print("Institution:", nwb.institution)
print("Lab:", nwb.lab)
print("Keywords:", nwb.keywords)

print("\nIntervals (keys):", list(nwb.intervals.keys()))
for k in nwb.intervals:
    interval = nwb.intervals[k]
    print(f"Interval '{k}': {len(interval['id'])} entries")

ecephys_module = nwb.processing.get('ecephys', None)
if ecephys_module:
    lfp = ecephys_module.data_interfaces.get('LFP', None)
    if lfp:
        print("\nLFP shape:", lfp.data.shape)
        print("LFP rate (Hz):", lfp.rate)
        print("LFP starting time (s):", lfp.starting_time)
        print("LFP num electrodes:", len(lfp.electrodes['id'].data[:]))
    else:
        print("\nNo LFP found in 'ecephys' module")
else:
    print("\nNo 'ecephys' processing module found")

electrodes = nwb.electrodes
print("\nNumber of electrodes:", len(electrodes['id'].data[:]))
print("Electrode table columns:", electrodes.colnames)

units = getattr(nwb, 'units', None)
if units:
    unit_ids = units['id'].data[:]
    print("\nNumber of sorted units:", len(unit_ids))
    print("Unit table columns:", units.colnames)
else:
    print("\nNo units found")