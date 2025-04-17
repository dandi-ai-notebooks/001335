"""
This script explores the basic structure and metadata of the NWB file,
focusing on the session description, blocks, and odor presentation timing.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic metadata
print("Session description:", nwb.session_description)
print("Identifier:", nwb.identifier)
print("Session start time:", nwb.session_start_time)
print("Experimenter:", nwb.experimenter)
print("Keywords:", nwb.keywords[:])
print("Lab:", nwb.lab)
print("Institution:", nwb.institution)

# Print subject information
print("\nSubject Information:")
print("Subject ID:", nwb.subject.subject_id)
print("Species:", nwb.subject.species)
print("Sex:", nwb.subject.sex)
print("Age:", nwb.subject.age)
print("Description:", nwb.subject.description)

# Get information about blocks
print("\nBlock Information:")
for block_name in ["Block 1", "Block 2", "Block 3"]:
    block = nwb.intervals[block_name]
    df = block.to_dataframe()
    start_time = df['start_time'].values[0]
    stop_time = df['stop_time'].values[0]
    duration = stop_time - start_time
    print(f"{block_name}: Start={start_time:.2f}s, Stop={stop_time:.2f}s, Duration={duration:.2f}s")
    print(f"Description: {block.description}")

# Get information about odor presentations
print("\nOdor Presentation Information:")
odor_intervals = {}
for odor in ["A", "B", "C", "D", "E", "F"]:
    interval_name = f"Odor {odor} ON"
    if interval_name in nwb.intervals:
        interval = nwb.intervals[interval_name]
        df = interval.to_dataframe()
        odor_intervals[odor] = df
        first_start = df['start_time'].iloc[0]
        last_stop = df['stop_time'].iloc[-1]
        avg_duration = (df['stop_time'] - df['start_time']).mean()
        n_presentations = len(df)
        print(f"Odor {odor}: {n_presentations} presentations, first at {first_start:.2f}s, last ending at {last_stop:.2f}s")
        print(f"Average presentation duration: {avg_duration:.2f}s")

# Create a histogram plot of odor presentation counts by time
plt.figure(figsize=(12, 6))
bin_width = 500  # bins of 500 seconds
max_time = 0

# Find the maximum time to set the plot range
for odor, df in odor_intervals.items():
    max_time = max(max_time, df['stop_time'].max())

bins = np.arange(0, max_time + bin_width, bin_width)

for i, (odor, df) in enumerate(odor_intervals.items()):
    # Use the midpoint of each presentation
    midpoints = (df['start_time'] + df['stop_time']) / 2
    plt.hist(midpoints, bins=bins, alpha=0.6, label=f'Odor {odor}')

plt.xlabel('Time (seconds)')
plt.ylabel('Number of Presentations')
plt.title('Odor Presentation Distribution Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('explore/odor_presentation_histogram.png')

# Create a plot of presentation times showing the temporal sequence
plt.figure(figsize=(12, 8))
plot_height = 0.8  # Height of each odor's bar

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
all_odors = sorted(odor_intervals.keys())

for i, odor in enumerate(all_odors):
    df = odor_intervals[odor]
    for j, row in df.iterrows():
        plt.plot([row['start_time'], row['stop_time']], 
                 [i + 1, i + 1], 
                 linewidth=6, 
                 solid_capstyle='butt',
                 color=colors[i % len(colors)])

# Add block boundaries as vertical lines
block_df = {block: nwb.intervals[block].to_dataframe() for block in ["Block 1", "Block 2", "Block 3"]}
for block, df in block_df.items():
    for _, row in df.iterrows():
        plt.axvline(row['start_time'], color='black', linestyle='--', alpha=0.5)
        plt.axvline(row['stop_time'], color='black', linestyle='--', alpha=0.5)
        # Add text label
        plt.text(row['start_time'] + (row['stop_time'] - row['start_time'])/2, 
                 len(all_odors) + 0.5, 
                 block, 
                 ha='center', 
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7))

plt.yticks(range(1, len(all_odors) + 1), [f'Odor {odor}' for odor in all_odors])
plt.xlabel('Time (seconds)')
plt.title('Odor Presentation Sequence')
plt.grid(True, alpha=0.3, axis='x')
plt.savefig('explore/odor_presentation_sequence.png')

# Create a third plot that zooms in on a smaller time period to see the pattern
# Take a 200 second window from the beginning of Block 1
zoom_start = block_df["Block 1"]['start_time'].values[0]
zoom_end = zoom_start + 200  # 200 second window

plt.figure(figsize=(12, 6))
for i, odor in enumerate(all_odors):
    df = odor_intervals[odor]
    filtered_df = df[(df['start_time'] >= zoom_start) & (df['stop_time'] <= zoom_end)]
    for j, row in filtered_df.iterrows():
        plt.plot([row['start_time'], row['stop_time']], 
                 [i + 1, i + 1], 
                 linewidth=6, 
                 solid_capstyle='butt',
                 color=colors[i % len(colors)])

plt.yticks(range(1, len(all_odors) + 1), [f'Odor {odor}' for odor in all_odors])
plt.xlabel('Time (seconds)')
plt.title(f'Odor Presentation Sequence (Zoomed: {zoom_start:.0f}s to {zoom_end:.0f}s)')
plt.xlim(zoom_start, zoom_end)
plt.grid(True, alpha=0.3, axis='x')
plt.savefig('explore/odor_presentation_sequence_zoomed.png')

print("\nExploration complete! Check the generated plots.")