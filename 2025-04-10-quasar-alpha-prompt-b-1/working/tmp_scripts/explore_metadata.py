# This script loads the NWB file and prints general metadata: session, subject info, available recordings, intervals, electrode and unit info.

import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"

file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

print("=== Session info ===")
print(f"Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Start time: {nwb.session_start_time}")
print(f"Keywords: {nwb.keywords[:]}")
print()

print("=== Subject info ===")
if nwb.subject:
    subject = nwb.subject
    print(f"ID: {subject.subject_id}")
    print(f"Species: {subject.species}")
    print(f"Sex: {subject.sex}")
    print(f"Age: {subject.age}")
    print(f"Description: {subject.description}")
else:
    print("No subject info found")

print()

print("=== Available intervals ===")
for int_name, interval in nwb.intervals.items():
    print(f"{int_name}: {interval.description} columns: {interval.colnames}")

print()

print("=== Processing modules and data interfaces ===")
for pname, pm in nwb.processing.items():
    print(f"Processing module: {pname} ({pm.description})")
    for diname, di in pm.data_interfaces.items():
        print(f"  DataInterface: {diname}, type: {type(di)}")

print()

print("=== Electrodes metadata ===")
etable = nwb.electrodes
print(f"Columns: {etable.colnames}")
print("First 5 electrode IDs:", etable.id[:5])

print()

print("=== Units (spikes) metadata ===")
units = nwb.units
print(f"Columns: {units.colnames}")
print("First 5 unit IDs:", units.id[:5])

io.close()