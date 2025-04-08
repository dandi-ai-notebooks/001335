# This script plots spike rasters for a few units aligned to odor A presentation onset times.
# The goal is to visualize event-related unit spiking.

import lindi
import pynwb
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set_theme()

lindi_url = "https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

odorA = nwb.intervals['Odor A ON']
odor_starts = np.array(odorA['start_time'][:])  # shape (101,)

units = nwb.units
num_units = len(units['id'].data)
np.random.seed(1)
indices = np.random.choice(num_units, size=min(10, num_units), replace=False)

plt.figure(figsize=(12, 6))

for i, unit_idx in enumerate(indices):
    spike_times = units['spike_times'][unit_idx]
    unit_id = units['id'].data[unit_idx]
    for trial_idx, t0 in enumerate(odor_starts):
        # Window around odor onset
        t_start = t0 - 0.5
        t_end = t0 + 1.5
        trial_spikes = spike_times[(spike_times >= t_start) & (spike_times <= t_end)] - t0
        plt.vlines(trial_spikes, i + trial_idx / 200, i + (trial_idx + 1)/200, color='k', linewidth=0.5)

plt.xlabel('Time relative to Odor A ON (s)')
plt.ylabel('Units')
plt.title('Raster plot: spike times aligned to Odor A ON')
plt.tight_layout()
plt.savefig('tmp_scripts/raster_odorA.png')
# plt.show() omitted