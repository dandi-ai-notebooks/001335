"""
This script explores the units (neural spiking) data in the NWB file, focusing on:
1. Unit properties and distribution
2. Spike times around odor presentations
3. Firing rate changes during odor presentation
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

print("Loading unit data...")
# Get information about units
print(f"Number of units: {len(nwb.units.id)}")
print(f"Unit columns: {nwb.units.colnames}")

# Create a DataFrame to store unit information
unit_data = {
    'unit_id': [],
    'electrode_group': [],
    'hemisphere': [],
    'depth': [],
    'n_spikes': []
}

# Extract unit data
print("\nExtracting unit data...")
for i in range(len(nwb.units.id)):
    unit_id = nwb.units.id[i]
    electrode_group = nwb.units['electrode_group'][i].name
    hemisphere = nwb.units['hemisphere'][i]
    depth = nwb.units['depth'][i]
    
    # Get spike times for this unit
    spike_times = nwb.units['spike_times'][i]
    n_spikes = len(spike_times)
    
    # Store in dictionary
    unit_data['unit_id'].append(unit_id)
    unit_data['electrode_group'].append(electrode_group)
    unit_data['hemisphere'].append(hemisphere)
    unit_data['depth'].append(depth) 
    unit_data['n_spikes'].append(n_spikes)

# Create DataFrame
units_df = pd.DataFrame(unit_data)

print("\nSample of unit data:")
print(units_df.head())

# Basic unit stats
print("\nUnits per electrode group:")
print(units_df.groupby('electrode_group')['unit_id'].count())

print("\nUnits per hemisphere:")
print(units_df.groupby('hemisphere')['unit_id'].count())

print("\nTotal recorded spikes by hemisphere:")
print(units_df.groupby('hemisphere')['n_spikes'].sum())

# Get information about odor presentation intervals
print("\nExtracting odor presentation intervals...")
odor_intervals = {}
for interval_name, interval in nwb.intervals.items():
    if "Odor" in interval_name and "ON" in interval_name:
        odor_name = interval_name.split(" ")[1]  # Extract just the odor letter
        odor_intervals[odor_name] = []
        for i in range(len(interval.id)):
            odor_intervals[odor_name].append((interval.start_time[i], interval.stop_time[i]))

print(f"Found intervals for {len(odor_intervals)} odors")

# Visualize unit depths by electrode group
plt.figure(figsize=(14, 8))
sns.boxplot(x='electrode_group', y='depth', data=units_df)
plt.title('Unit Depths by Electrode Group')
plt.xlabel('Electrode Group')
plt.ylabel('Depth (μm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("tmp_scripts/unit_depths.png")

# Visualize unit count by electrode group
plt.figure(figsize=(10, 6))
unit_counts = units_df['electrode_group'].value_counts().sort_index()
plt.bar(unit_counts.index, unit_counts.values)
plt.title('Number of Units per Electrode Group')
plt.xlabel('Electrode Group')
plt.ylabel('Number of Units')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("tmp_scripts/unit_counts.png")

# Calculate firing rates for each unit
print("\nCalculating average firing rates...")
# Using entire recording duration for overall rate
recording_duration = nwb.processing["ecephys"].data_interfaces["LFP"].data.shape[0] / nwb.processing["ecephys"].data_interfaces["LFP"].rate
units_df['firing_rate'] = units_df['n_spikes'] / recording_duration

# Create a histogram of firing rates
plt.figure(figsize=(10, 6))
bin_max = min(50, units_df['firing_rate'].max() + 5)  # Cap at 50 Hz for better visualization
plt.hist(units_df['firing_rate'], bins=np.linspace(0, bin_max, 50), alpha=0.7, color='skyblue')
plt.title('Distribution of Unit Firing Rates')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Number of Units')
plt.axvline(units_df['firing_rate'].median(), color='red', linestyle='--', 
            label=f'Median: {units_df["firing_rate"].median():.2f} Hz')
plt.legend()
plt.tight_layout()
plt.savefig("tmp_scripts/firing_rate_histogram.png")

# Analyze spike times around odor presentations (peristimulus time histograms)
print("\nAnalyzing spike times around odor presentations...")

# Choose an odor for PSTH
target_odor = "A"
if target_odor in odor_intervals:
    # Parameters for PSTH
    window_before = 2  # seconds before odor onset
    window_after = 3   # seconds after odor onset
    bin_width = 0.1    # seconds per bin
    
    # Create time bins for histogram
    bins = np.arange(-window_before, window_after + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    
    # Select a subset of units for analysis (top 20 most active)
    top_units = units_df.sort_values('n_spikes', ascending=False).head(20)
    
    # Create a matrix to hold PSTH data
    psth_matrix = np.zeros((len(top_units), len(bins)-1))
    
    # Loop through units and compile spike times relative to odor onset
    for i, (idx, unit) in enumerate(top_units.iterrows()):
        unit_id = unit['unit_id']
        spike_times = nwb.units['spike_times'][unit_id]
        
        # Compile spikes for all presentations of this odor
        all_presentation_spikes = []
        
        for onset, offset in odor_intervals[target_odor]:
            # Extract spikes around this presentation
            window_start = onset - window_before
            window_end = onset + window_after
            
            # Find spikes within this window
            mask = (spike_times >= window_start) & (spike_times <= window_end)
            presentation_spikes = spike_times[mask]
            
            # Convert to time relative to odor onset
            presentation_spikes = presentation_spikes - onset
            
            all_presentation_spikes.extend(presentation_spikes)
            
        # Create histogram
        hist, _ = np.histogram(all_presentation_spikes, bins=bins)
        
        # Normalize by number of presentations and bin width to get firing rate
        psth_matrix[i, :] = hist / (len(odor_intervals[target_odor]) * bin_width)
    
    # Plot PSTH heatmap
    plt.figure(figsize=(12, 10))
    
    # Create custom colormap (from white to blue)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'darkblue'])
    
    # Plot heatmap
    plt.imshow(psth_matrix, aspect='auto', cmap=cmap, 
               extent=[-window_before, window_after, 0, len(top_units)])
    
    # Add a vertical line at odor onset (t=0)
    plt.axvline(x=0, color='red', linestyle='--', label='Odor Onset')
    
    # Calculate the typical odor offset time and add a vertical line
    # Assuming all presentations are the same duration
    odor_duration = odor_intervals[target_odor][0][1] - odor_intervals[target_odor][0][0]
    plt.axvline(x=odor_duration, color='blue', linestyle='--', label='Odor Offset')
    
    # Format plot
    plt.colorbar(label='Firing Rate (Hz)')
    plt.title(f'PSTH for Odor {target_odor} (Top 20 Units)')
    plt.xlabel('Time from Odor Onset (s)')
    plt.ylabel('Unit #')
    plt.legend()
    plt.tight_layout()
    plt.savefig("tmp_scripts/odor_psth_heatmap.png")
    
    # Plot average PSTH across all units
    plt.figure(figsize=(12, 6))
    mean_psth = np.mean(psth_matrix, axis=0)
    sem_psth = np.std(psth_matrix, axis=0) / np.sqrt(len(top_units))
    
    # Plot mean with error bars
    plt.fill_between(bin_centers, mean_psth - sem_psth, mean_psth + sem_psth, alpha=0.3, color='skyblue')
    plt.plot(bin_centers, mean_psth, color='blue')
    
    # Add vertical lines
    plt.axvline(x=0, color='red', linestyle='--', label='Odor Onset')
    plt.axvline(x=odor_duration, color='blue', linestyle='--', label='Odor Offset')
    
    # Format plot
    plt.title(f'Average PSTH for Odor {target_odor} (Top 20 Units)')
    plt.xlabel('Time from Odor Onset (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tmp_scripts/average_odor_psth.png")
    
    # Calculate and visualize the change in firing rate during odor presentation
    print("\nCalculating firing rate changes during odor presentation...")
    
    # For each unit, calculate baseline and odor-evoked firing rates
    rate_changes = []
    
    for idx, unit in top_units.iterrows():
        unit_id = unit['unit_id']
        spike_times = nwb.units['spike_times'][unit_id]
        
        baseline_rates = []
        odor_rates = []
        
        for onset, offset in odor_intervals[target_odor]:
            # Baseline period: before odor
            baseline_start = onset - window_before
            baseline_end = onset - 0.2  # Slight buffer before odor onset
            
            # Find spikes within baseline window
            baseline_mask = (spike_times >= baseline_start) & (spike_times <= baseline_end)
            n_baseline_spikes = np.sum(baseline_mask)
            baseline_duration = baseline_end - baseline_start
            baseline_rate = n_baseline_spikes / baseline_duration
            baseline_rates.append(baseline_rate)
            
            # Odor period
            odor_start = onset
            odor_end = offset
            
            # Find spikes within odor window
            odor_mask = (spike_times >= odor_start) & (spike_times <= odor_end)
            n_odor_spikes = np.sum(odor_mask)
            odor_duration = odor_end - odor_start
            odor_rate = n_odor_spikes / odor_duration
            odor_rates.append(odor_rate)
        
        # Calculate average rates
        avg_baseline = np.mean(baseline_rates)
        avg_odor = np.mean(odor_rates)
        
        # Store results
        rate_changes.append({
            'unit_id': unit_id,
            'baseline_rate': avg_baseline,
            'odor_rate': avg_odor,
            'percent_change': (avg_odor - avg_baseline) / avg_baseline * 100 if avg_baseline > 0 else 0,
            'electrode_group': unit['electrode_group'],
            'hemisphere': unit['hemisphere']
        })
    
    # Convert to DataFrame
    rate_change_df = pd.DataFrame(rate_changes)
    
    # Create a paired plot showing baseline vs. odor firing rates
    plt.figure(figsize=(10, 8))
    
    # Plot points
    for i in range(len(rate_change_df)):
        x = rate_change_df.iloc[i]['baseline_rate']
        y = rate_change_df.iloc[i]['odor_rate']
        plt.plot([x, y], [i, i], 'k-', alpha=0.3)
        plt.plot(x, i, 'o', color='blue', alpha=0.7, label='Baseline' if i == 0 else "")
        plt.plot(y, i, 'o', color='red', alpha=0.7, label='Odor' if i == 0 else "")
    
    # Format plot
    plt.title(f'Firing Rate Changes During Odor {target_odor} Presentation')
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('Unit #')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("tmp_scripts/odor_rate_changes.png")
    
    # Visualize percent change in firing rate
    plt.figure(figsize=(12, 6))
    
    # Sort by percent change
    sorted_df = rate_change_df.sort_values('percent_change')
    
    # Create a horizontal bar plot
    plt.barh(np.arange(len(sorted_df)), sorted_df['percent_change'], color='skyblue')
    
    # Add lines for reference
    plt.axvline(x=0, color='black', linestyle='-')
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=-50, color='blue', linestyle='--', alpha=0.5)
    
    # Format plot
    plt.title(f'Percent Change in Firing Rate During Odor {target_odor} Presentation')
    plt.xlabel('Percent Change from Baseline (%)')
    plt.ylabel('Units (Sorted)')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("tmp_scripts/odor_percent_changes.png")
    
else:
    print(f"Odor {target_odor} not found in intervals")