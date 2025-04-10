# %% [markdown]
# # AI-generated exploratory notebook for Dandiset 001335
# 
# **DISCLAIMER**: This notebook was auto-generated using `dandi-notebook-gen` (AI) and **has not been fully verified**. Use caution when interpreting code or results, and consider this notebook primarily as a starting point for your analyses.
# 
# ---
# 
# # Dandiset 001335: Neuropixels Recordings from Hippocampus of head-fixed mice during odor presentation
# 
# **Description**: Head-fixed wild type mice were presented with various odor sequences, as neural activity was recorded from hippocampus using Neuropixels probes.
# 
# **Contributors**: Mohapatra, Manish; Halchenko, Yaroslav
# 
# **Date Created**: 2025-02-14
# 
# **License**: CC-BY-4.0
# 
# **Citation**:
# *Mohapatra, Manish; Halchenko, Yaroslav (2025) Neuropixels Recordings from Hippocampus of head-fixed mice during odor presentation (Version draft) [Data set]. DANDI Archive.* https://dandiarchive.org/dandiset/001335/draft
# 
# ---

# %% [markdown]
# ## 1. Setup
# This notebook relies on PyNWB, h5py, remfile, matplotlib, seaborn, and the DANDI Python client.
# 
# Please ensure they are installed (`pip install pynwb h5py remfile matplotlib seaborn dandi`)
# 
# Imports and some matplotlib style:

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pynwb
import h5py
import remfile
from dandi.dandiapi import DandiAPIClient

sns.set_theme()

# %% [markdown]
# ## 2. List all assets in Dandiset via DANDI API

# %%
client = DandiAPIClient()
dandiset = client.get_dandiset("001335")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in Dandiset 001335:")
for asset in assets:
    print(asset.path)

# %% [markdown]
# This notebook will focus on the NWB file:
# 
# **`sub-M541/sub-M541_ecephys.nwb`**

# %% [markdown]
# ## 3. Load NWB file remotely
# The following code streams the NWB file using remfile, h5py, and PyNWB.

# %%
nwb_url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"

fileobj = remfile.File(nwb_url)
f = h5py.File(fileobj)
io = pynwb.NWBHDF5IO(file=f, load_namespaces=True)
nwbfile = io.read()

print("Loaded NWB file:")
print("Identifier:", nwbfile.identifier)
print("Session start time:", nwbfile.session_start_time)
print("Session description:", nwbfile.session_description)
print("Subject ID:", nwbfile.subject.subject_id)
print("Subject species:", nwbfile.subject.species)

# %% [markdown]
# ## 4. Explore session metadata, subject, and experiment setup

# %%
print("Experimenter(s):", nwbfile.experimenter)
print("Experiment description:", nwbfile.experiment_description)
print("Institution:", nwbfile.institution)
print("Lab:", nwbfile.lab)

subject = nwbfile.subject
print("Subject Description:", subject.description)
print("Sex:", subject.sex)
print("Age:", subject.age)

# %% [markdown]
# ## 5. Explore electrode groups

# %%
for group_name, group in nwbfile.electrode_groups.items():
    print(f"Group: {group_name}")
    print(" - Location:", group.location)
    print(" - Device:", group.device.name)
    print(" - Description:", group.description)
    print("")

# %% [markdown]
# ## 6. Explore extracellular electrodes metadata

# %%
electrodes = nwbfile.electrodes
print("Electrode table columns:", electrodes.colnames)
elec_df = electrodes.to_dataframe()
elec_df.head()

# %% [markdown]
# Visualization: Electrode depth distribution

# %%
plt.figure(figsize=(10, 4))
sns.histplot(elec_df['depth'], bins=30, kde=True)
plt.xlabel('Electrode depth (µm)')
plt.ylabel('Count')
plt.title('Distribution of Electrode Depths')
plt.show()

# %% [markdown]
# ## 7. Explore LFP data summary and quick visualization
# This dataset contains LFP recordings with shape approximately (15 million samples × 64 channels, ~2500 Hz sampling).
# 
# To avoid large downloads and long wait times, we will fetch **first ~2 seconds** of data from a **few channels** as an illustration.

# %%
lfp = nwbfile.processing['ecephys'].data_interfaces['LFP']
rate = lfp.rate  # 2500 Hz
snippet_duration_seconds = 2
snippet_samples = int(rate * snippet_duration_seconds)

data = lfp.data

snippet = data[:snippet_samples, :10]  # first 2 seconds, first 10 channels
time = np.arange(snippet_samples) / rate

plt.figure(figsize=(12, 6))
offset = 0
for i in range(snippet.shape[1]):
    plt.plot(time, snippet[:, i] + offset, label=f'Ch {i}')
    offset += np.ptp(snippet[:, i]) * 1.2  # vertical offset between traces
plt.xlabel('Time (s)')
plt.ylabel('Amplitude + offset (V)')
plt.title('Example LFP traces (first 2s, 10 channels)')
plt.legend()
plt.show()

# %% [markdown]
# ## 8. Explore experimental intervals (odor/presentation blocks)
# The .intervals attribute holds behavioral and experimental event timing annotations.

# %%
for name, tbl in nwbfile.intervals.items():
    print(f"Interval: {name}")
    print("  Description:", tbl.description)
    print("  Columns:", tbl.colnames)

# %% [markdown]
# Plot: Timeline of odor block intervals

# %%
block_names = [k for k in nwbfile.intervals.keys() if 'Block' in k]
fig, ax = plt.subplots(figsize=(12, 2 + len(block_names)))
for idx, name in enumerate(block_names):
    tbl = nwbfile.intervals[name]
    df = tbl.to_dataframe()
    for _, row in df.iterrows():
        ax.plot([row['start_time'], row['stop_time']], [idx, idx], lw=4)
ax.set_yticks(range(len(block_names)))
ax.set_yticklabels(block_names)
ax.set_xlabel('Time (s)')
ax.set_title('Experimental Blocks Timeline')
plt.show()

# %% [markdown]
# ## 9. Units table overview

# %%
units = nwbfile.units
print("Units table columns:", units.colnames)
units_df = units.to_dataframe()

units_df.head()

# %% [markdown]
# Plot: Distribution of spike counts per unit (limited to first 50 units for speed)

# %%
subset_units = units_df.head(50)
subset_units['n_spikes'] = subset_units['spike_times'].apply(lambda x: len(x))
plt.figure(figsize=(10,4))
sns.histplot(subset_units['n_spikes'], bins=20)
plt.xlabel('Spike count')
plt.ylabel('Unit count')
plt.title('Spike Count Distribution (First 50 Units)')
plt.show()

# %% [markdown]
# # End of notebook
# This notebook serves as an initial exploration. You are encouraged to customize it and perform deeper analyses relevant to your research questions.