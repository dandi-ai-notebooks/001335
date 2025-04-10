# %% [markdown]
# **NOTE: This automated notebook was AI-generated using dandi-notebook-gen. Please exercise caution when interpreting the code or scientific results. Manual review is strongly recommended.**
# 
# # DANDI Dandiset 001335: Neuropixels Recordings from Hippocampus during Odor Presentation
#
# This tutorial demonstrates how to explore and analyze data from an openly shared neurophysiology dataset.
#
# **Citation:** Mohapatra, Manish; Halchenko, Yaroslav (2025). Neuropixels Recordings from Hippocampus of head-fixed mice during odor presentation (Version draft). DANDI Archive. https://dandiarchive.org/dandiset/001335/draft
#
# ## Dandiset Description
# Head-fixed wild type male mice (~8 weeks old) were presented with various odor sequences while neural activity was recorded using Neuropixels probes targeting dorsal CA1.
#
# The available dataset contains LFP and spike data, with annotations for different odor presentation blocks.
#
# ---
#
# ## Import necessary packages
# (Please ensure the following packages are installed: `pynwb`, `remfile`, `h5py`, `dandi`, `numpy`, `matplotlib`)

# %%
import matplotlib.pyplot as plt
import numpy as np
import pynwb
import remfile
import h5py
from dandi.dandiapi import DandiAPIClient

# %% [markdown]
# ## Accessing Dandiset metadata through the DANDI API

# %%
client = DandiAPIClient()
dandiset = client.get_dandiset("001335", "draft")
assets = list(dandiset.get_assets())
print(f"Number of assets in Dandiset: {len(assets)}")
for asset in assets:
    print(asset.path)

# %% [markdown]
# ## Load the NWB file remotely using PyNWB and remfile
# 
# We are using the main NWB file `sub-M541/sub-M541_ecephys.nwb`.

# %%
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"

rf = remfile.File(url)
hf = h5py.File(rf)
io = pynwb.NWBHDF5IO(file=hf, load_namespaces=True)
nwb = io.read()

# %% [markdown]
# ## Explore session and subject metadata

# %%
print(f"Session description: {nwb.session_description}")
print(f"Experiment ID: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experiment keywords: {nwb.keywords[:]}")
if nwb.subject:
    print(f"Subject ID: {nwb.subject.subject_id}")
    print(f"Species: {nwb.subject.species}")
    print(f"Sex: {nwb.subject.sex}")
    print(f"Age: {nwb.subject.age}")
    print(f"Subject description: {nwb.subject.description}")

# %% [markdown]
# ## Experimental blocks and odor intervals

# %%
print("All intervals/time segments:")
for name, interval in nwb.intervals.items():
    print(f"{name}: {interval.description} columns={interval.colnames}")

# %% [markdown]
# ## Channels/electrodes table

# %%
etable = nwb.electrodes
print("Electrode table columns:", etable.colnames)
print("First 5 electrode IDs:", etable.id[:5])

# %% [markdown]
# ## Spike Units Table

# %%
units = nwb.units
print("Spike unit columns:", units.colnames)
print("First 5 unit IDs:", units.id[:5])

# %% [markdown]
# ## Plot spike raster of 10 units over first 5 seconds
# Here we visualize the spiking activity of the first 10 identified units.

# %%
unit_ids = list(units.id[:])
max_units = min(10, len(unit_ids))
time_window = 5  # seconds

plt.figure(figsize=(10, 6))

for idx in range(max_units):
    unit_id = unit_ids[idx]
    spike_times = units['spike_times'][idx]
    mask = (spike_times >= 0) & (spike_times <= time_window)
    plt.vlines(spike_times[mask], idx + 0.5, idx + 1.5)

plt.xlabel('Time (s)')
plt.ylabel('Unit ID')
plt.yticks(np.arange(1, max_units + 1), unit_ids[:max_units])
plt.title(f'Spike raster: first {max_units} units (first {time_window} seconds)')

plt.tight_layout()
plt.show()

# %% [markdown]
# 
# ## Notes
# 
# - This notebook illustrates how to load metadata, spike times, and basic interval information from the NWB file without downloading it entirely.
# - Because raw LFP data may be large and noisy/artifactual in this dataset, analysis focuses on sorted units here.
# - For your own analyses, consider segmenting by odor blocks, comparing spiking or LFP characteristics, or extracting aligned data snippets.
# - Extensive computational analyses such as spike sorting or LFP filtering are beyond this example scope.
# 
# ---
# 
# **End of AI-generated example analysis notebook**.