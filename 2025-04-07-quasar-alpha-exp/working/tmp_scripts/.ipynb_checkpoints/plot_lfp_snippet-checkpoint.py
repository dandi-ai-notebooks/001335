# This script loads part of the LFP data from the NWB file and generates a plot snippet for several channels.
# It visualizes a short segment of LFP signals across different electrode depths.

import lindi
import pynwb
import matplotlib.pyplot as plt
import numpy as np

# Enable nice style
import seaborn as sns
sns.set_theme()

lindi_url = "https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

lfp = nwb.processing['ecephys']['LFP']

# Time window
rate = lfp.rate  # 2500 Hz
duration_sec = 5
n_samples = int(rate * duration_sec)  # 12500 samples

# Select subset of channels
num_channels = lfp.data.shape[1]
channel_indices = np.linspace(0, num_channels - 1, 8, dtype=int)

# Load data subset
snippet = lfp.data[:n_samples, :][:, channel_indices]

time = np.arange(n_samples) / rate

plt.figure(figsize=(12, 6))

offset = 0
offsets = []
for ch_idx in range(snippet.shape[1]):
    snippet_channel = snippet[:, ch_idx]
    plt.plot(time, snippet_channel + offset, label=f'Ch {channel_indices[ch_idx]}')
    offsets.append(offset)
    offset += np.ptp(snippet_channel) * 1.2  # vertical offset based on signal amplitude

plt.xlabel('Time (s)')
plt.ylabel('LFP + offset')
plt.title('LFP snippet (~5 seconds) from selected channels')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_snippet.png')
# plt.show()  # Omitted to avoid hanging