# This script explores annotated time intervals in the NWB file,
# summarizing stimulus presentations (odors) and experimental blocks.

import remfile
import h5py
import pynwb

url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"

print("Opening NWB file...")
file = remfile.File(url)
f = h5py.File(file, 'r')
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

print("Intervals in NWB:")

for interval_name in nwb.intervals:
    interval_table = nwb.intervals[interval_name]
    print(f"- {interval_name}: {interval_table.description}")
    try:
        nrows = len(interval_table.id)
        print(f"  Number of intervals: {nrows}")
        durations = []
        for ind in range(nrows):
            start = interval_table['start_time'][ind]
            stop = interval_table['stop_time'][ind]
            durations.append(stop - start)
        if durations:
            print(f"  Interval durations (sec): min={min(durations):.3f}, max={max(durations):.3f}, mean={sum(durations)/len(durations):.3f}")
    except Exception as e:
        print(f"  Error retrieving interval info: {str(e)}")

io.close()