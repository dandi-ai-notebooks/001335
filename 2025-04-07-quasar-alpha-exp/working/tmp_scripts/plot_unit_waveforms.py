# This script loads the spike waveform means from a subsample of units and plots them.
# This explores spike shape variability across sorted units in the NWB file.

import lindi
import pynwb
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set_theme()

lindi_url = "https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

units = nwb.units

num_units = len(units['id'].data)
np.random.seed(0)
indices = np.random.choice(num_units, size=min(15, num_units), replace=False)

waveforms = units['waveform_mean'].data[indices]

plt.figure(figsize=(12, 6))

for i, idx in enumerate(indices):
    plt.plot(waveforms[i], label=f'Unit ID {units["id"].data[idx]}')

plt.xlabel('Sample index (~us)')
plt.ylabel('Amplitude')
plt.title('Mean spike waveforms of sampled units')
plt.legend(fontsize='small', ncol=3, loc='upper right')
plt.tight_layout()
plt.savefig('tmp_scripts/unit_waveforms.png')
# plt.show() omitted