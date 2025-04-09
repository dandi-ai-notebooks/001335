"""
This script explores the electrode information in the NWB file and visualizes their attributes.
We'll examine:
1. Electrode locations and depths
2. Distribution of electrodes across shanks
3. Electrode group properties
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Extract electrode data into a pandas DataFrame
electrode_data = {
    'id': [],
    'location': [],
    'group_name': [], 
    'depth': [],
    'hemisphere': []
}

print("Electrode column names:", nwb.electrodes.colnames)

# Collect electrode data
for i in range(len(nwb.electrodes.id)):
    electrode_data['id'].append(i)
    electrode_data['location'].append(nwb.electrodes['location'][i])
    electrode_data['group_name'].append(nwb.electrodes['group_name'][i])
    electrode_data['depth'].append(nwb.electrodes['depth'][i])
    electrode_data['hemisphere'].append(nwb.electrodes['hemisphere'][i])

electrodes_df = pd.DataFrame(electrode_data)

print("\nElectrodes overview:")
print(f"Number of electrodes: {len(electrodes_df)}")
print("\nSample of electrode data:")
print(electrodes_df.head())

# Count electrodes by group
group_counts = electrodes_df['group_name'].value_counts().sort_index()
print("\nNumber of electrodes per group:")
print(group_counts)

# Count electrodes by location
location_counts = electrodes_df['location'].value_counts()
print("\nNumber of electrodes per location:")
print(location_counts)

# Count electrodes by hemisphere
hemisphere_counts = electrodes_df['hemisphere'].value_counts()
print("\nNumber of electrodes per hemisphere:")
print(hemisphere_counts)

# Visualize electrode depth distribution by group
plt.figure(figsize=(10, 8))
groups = sorted(electrodes_df['group_name'].unique())

# Create a bar plot of electrode count by group
plt.subplot(2, 1, 1)
plt.bar(group_counts.index, group_counts.values)
plt.title('Number of Electrodes per Group')
plt.xlabel('Group Name')
plt.ylabel('Number of Electrodes')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('tmp_scripts/electrode_group_counts.png')

# Create a plot of electrode depths by group
plt.figure(figsize=(12, 8))
for i, group in enumerate(groups):
    group_data = electrodes_df[electrodes_df['group_name'] == group]
    x_positions = np.ones(len(group_data)) * i
    plt.scatter(x_positions, group_data['depth'], alpha=0.5, 
                label=f"{group} (n={len(group_data)})")

plt.title('Electrode Depths by Group')
plt.xlabel('Group')
plt.xticks(range(len(groups)), groups, rotation=45)
plt.ylabel('Depth (μm)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('tmp_scripts/electrode_depths_by_group.png')

# Visualize electrode placement
# Group by probe (imec0 vs imec1) and shank (0-3)
plt.figure(figsize=(12, 8))

# Get probe names
probes = sorted(list(set([group_name.split('.')[0] for group_name in groups])))

for p_idx, probe in enumerate(probes):
    plt.subplot(1, len(probes), p_idx + 1)
    
    # Filter for this probe
    probe_mask = electrodes_df['group_name'].str.contains(probe)
    probe_df = electrodes_df[probe_mask]
    
    # Get shanks for this probe
    shanks = sorted(list(set([group_name.split('.')[1] for group_name in probe_df['group_name']])))
    
    # Plot electrodes by shank
    for s_idx, shank in enumerate(shanks):
        shank_mask = probe_df['group_name'].str.contains(f"{probe}.{shank}")
        shank_df = probe_df[shank_mask]
        
        # Calculate x-position for this shank (evenly space shanks horizontally)
        x_pos = s_idx
        
        # Plot points for this shank
        plt.scatter(x_pos * np.ones(len(shank_df)), shank_df['depth'], 
                   alpha=0.5, label=f"shank{shank[-1]}")
    
    plt.title(f"Probe {probe}")
    plt.xlabel("Shank")
    plt.xticks(range(len(shanks)), [f"shank{shank[-1]}" for shank in shanks])
    plt.ylabel("Depth (μm)")
    plt.ylim([electrodes_df['depth'].min() - 100, electrodes_df['depth'].max() + 100])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

plt.tight_layout()
plt.savefig('tmp_scripts/electrode_placement.png')

# Visualize the distribution of electrodes across hemispheres
plt.figure(figsize=(8, 6))
electrode_counts = hemisphere_counts.values
labels = hemisphere_counts.index

plt.pie(electrode_counts, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=['skyblue', 'lightcoral'])
plt.title('Electrode Distribution by Hemisphere')
plt.tight_layout()
plt.savefig('tmp_scripts/electrode_hemisphere_distribution.png')

# Print summary statistics on electrode depths
print("\nElectrode depth statistics (μm):")
print(electrodes_df['depth'].describe())