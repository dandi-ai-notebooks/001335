"""
This script explores the basic structure of the NWB file in Dandiset 001335,
focusing on:
1. Basic metadata
2. Electrode information
3. LFP data structure
4. Odor presentation intervals
5. Available units (neural activity)
"""

import pynwb
import h5py
import remfile
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Print basic information about the dataset
print("=" * 50)
print("BASIC METADATA")
print("=" * 50)
print(f"Session ID: {nwb.identifier}")
print(f"Description: {nwb.session_description}")
print(f"Experiment Description: {nwb.experiment_description}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}")

# Print subject information
print("\n" + "=" * 50)
print("SUBJECT INFORMATION")
print("=" * 50)
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")
print(f"Description: {nwb.subject.description}")

# Print electrode group information
print("\n" + "=" * 50)
print("ELECTRODE GROUPS")
print("=" * 50)
for group_name, group in nwb.electrode_groups.items():
    print(f"Group: {group_name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device: {group.device.description} (Manufacturer: {group.device.manufacturer})")

# Print LFP data information
print("\n" + "=" * 50)
print("LFP DATA")
print("=" * 50)
lfp = nwb.processing["ecephys"].data_interfaces["LFP"]
print(f"LFP Description: {lfp.description}")
print(f"LFP Unit: {lfp.unit}")
print(f"Sampling Rate: {lfp.rate} Hz")
print(f"Data Shape: {lfp.data.shape} (Samples × Channels)")
print(f"Total Duration: {lfp.data.shape[0] / lfp.rate:.2f} seconds ({lfp.data.shape[0] / lfp.rate / 60:.2f} minutes)")

# Print information about odor presentation intervals
print("\n" + "=" * 50)
print("ODOR PRESENTATION INTERVALS")
print("=" * 50)
for interval_name, interval in nwb.intervals.items():
    if "Block" in interval_name:
        print(f"Block: {interval_name}")
        # Get all intervals for this block
        intervals_array = []
        for i in range(len(interval.id)):
            intervals_array.append((interval.start_time[i], interval.stop_time[i]))
        if intervals_array:
            start_times, stop_times = zip(*intervals_array)
            block_duration = sum(stop - start for start, stop in intervals_array)
            print(f"  Block Start Time: {min(start_times):.2f} seconds")
            print(f"  Block End Time: {max(stop_times):.2f} seconds")
            print(f"  Block Duration: {block_duration:.2f} seconds")
            print(f"  Number of Intervals: {len(intervals_array)}")
    elif "Odor" in interval_name:
        print(f"Odor: {interval_name}")
        # Get all intervals for this odor
        intervals_array = []
        for i in range(len(interval.id)):
            intervals_array.append((interval.start_time[i], interval.stop_time[i]))
        if intervals_array:
            odor_total_duration = sum(stop - start for start, stop in intervals_array)
            print(f"  Number of Presentations: {len(intervals_array)}")
            print(f"  Total Duration: {odor_total_duration:.2f} seconds")
            if intervals_array:
                presentation_durations = [stop - start for start, stop in intervals_array]
                print(f"  Mean Presentation Duration: {np.mean(presentation_durations):.2f} seconds")

# Print information about units (neural activity)
print("\n" + "=" * 50)
print("UNITS (NEURAL ACTIVITY)")
print("=" * 50)
print(f"Number of Units: {len(nwb.units.id)}")
print(f"Unit Column Fields: {nwb.units.colnames}")

# Count units per electrode group
if 'electrode_group' in nwb.units.colnames:
    group_counts = {}
    for i in range(len(nwb.units.id)):
        group = nwb.units['electrode_group'][i]
        group_name = group.name
        if group_name not in group_counts:
            group_counts[group_name] = 0
        group_counts[group_name] += 1
    
    print("\nUnits per Electrode Group:")
    for group_name, count in group_counts.items():
        print(f"  {group_name}: {count} units")