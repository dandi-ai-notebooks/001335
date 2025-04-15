# %% [markdown]
# # Exploring Dandiset 001335: Neuropixels Recordings from Hippocampus of head-fixed mice during odor presentation

# %% [markdown]
# **NOTE**: This notebook was automatically generated using AI (dandi-notebook-gen) and has not been fully verified. Please exercise caution when interpreting the code or results. 

# %% [markdown]
# ## Dandiset Overview
# The dataset contains Neuropixels recordings from the hippocampus of head-fixed wild-type mice during odor presentation. The contributor of this dataset includes Mohapatra, Manish, and Halchenko, Yaroslav.
# 
# DANDI ID: 001335/draft  
# License: CC-BY-4.0  
# Link: [View on DANDI](https://dandiarchive.org/dandiset/001335/draft)  
# Description: Head-fixed wild type mice were presented with various odor sequences, while neural activity was recorded from the hippocampus using Neuropixels probes.

# %% [markdown]
# ## Notebook Overview
# This notebook will cover the following:
# - Loading the Dandiset using the DANDI API.
# - Exploring metadata of NWB files.
# - Visualizing neuronal data such as Local Field Potentials (LFP).
# - Possible directions for future analysis.

# %% [markdown]
# ## Required Packages
# Ensure the following packages are available in your Python environment:
# - `dandi`
# - `pynwb`
# - `h5py`
# - `numpy`
# - `matplotlib`
# - `pandas`
# - `remfile`

# %% [markdown]
# ## Connecting to DANDI API and Loading the Dataset
from dandi.dandiapi import DandiAPIClient
import remfile
import h5py
import pynwb
import numpy as np
import matplotlib.pyplot as plt

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001335", "draft")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading Metadata from NWB File
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Experimenter: {nwb.experimenter}")
print(f"Keywords: {nwb.keywords[:]}")

# %% [markdown]
# ## Data Visualization: Local Field Potentials (LFP)
# Extracting and visualizing LFP data from the NWB file.

lfp_data = nwb.processing["ecephys"].data_interfaces["LFP"].data[0:1000, 0:10] # First 1000 samples of first 10 electrodes
time_series = np.arange(0, lfp_data.shape[0], 1) / nwb.processing["ecephys"].data_interfaces["LFP"].rate

plt.figure(figsize=(15, 5))
plt.plot(time_series, lfp_data)
plt.title("LFP Data Sample")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.show()

# %% [markdown]
# ## Summary and Future Directions
# In this notebook, we successfully loaded, explored, and visualized a subset of LFP data from the NWB file in Dandiset 001335. In future analysis, further exploration of the data, including firing rates, connectivity analysis, and cross-population correlations, could provide valuable insights into neural activity behaviors.

# Closing the file
io.close()