
from __future__ import annotations
import matplotlib.pyplot as plt
import pynapple as nap
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import json
import pandas as pd
from dateutil import parser
from warnings import warn
from spikeinterface.core import SortingAnalyzer, BaseSorting


def get_epochs(recording):
    """
    Returns a DataFrame with the start and end times (in seconds) of each epoch in the recording.

    Parameters
    ----------
    recording : spikeinterface.Recording
        The recording object to extract epochs from.

    Returns
    -------
    epoch_df : pd.DataFrame
        A DataFrame with columns 'start' and 'end' representing the epoch start and end times (in seconds).
    """

    # Get epoch start and end times in samples
    epoch_starts = [0]
    epoch_ends = [recording.get_num_samples(0)]

    for epoch in range(recording.get_num_segments() - 1):
        epoch_starts.append(recording.get_num_samples(epoch) + epoch_starts[-1])
        epoch_ends.append(recording.get_num_samples(epoch + 1) + epoch_ends[-1] - 1)

    # Convert from samples to seconds
    epoch_starts = np.array(epoch_starts) / recording.get_sampling_frequency()
    epoch_ends = np.array(epoch_ends) / recording.get_sampling_frequency()

    # Create a DataFrame
    epochs = pd.DataFrame({
        'start': epoch_starts,
        'end': epoch_ends
    })

    return epochs

def add_neuroscope_mapping(probe, xml_channel_indices):
    # Sanity check for number of channels
    if probe.get_contact_count() != len(xml_channel_indices):
        print("Number of Neuroscope channels doesn't match the SpikeInterface probe!")  # Sanity check for xml channel indices
        print("Check Neuroscope mapping")  # Sanity check for xml channel indices

    # Sort contact positions based on shank IDs and y-coordinates
    shank_ids = probe.shank_ids
    contact_positions = probe.contact_positions
    unique_shank_ids = np.unique(shank_ids)

    # sort channels based on their coordinates, top to bottom and shank-wise
    sorted_coordinates_by_shank = []
    for unique_id in unique_shank_ids:
        id_indices = np.where(shank_ids == unique_id)[0]
        coors = contact_positions[id_indices]
        coors = coors[coors[:, 1].argsort()[::-1]]
        sorted_coordinates_by_shank.append(coors)
    final_sorted_coordinates = np.vstack(sorted_coordinates_by_shank)

    # Update probe with sorted coordinates
    probe.set_contacts(
        final_sorted_coordinates,
        shapes=probe.contact_shapes,
        shape_params=probe.contact_shape_params,
        plane_axes=probe.contact_plane_axes,
        contact_ids=np.arange(len(xml_channel_indices)),
        shank_ids=np.sort(probe.shank_ids),  # Sorted shank_ids
    )

    # Set device channel indices
    probe.set_device_channel_indices(xml_channel_indices)

    print("Probe updated with Neuroscope mapping")

    return probe



def read_xml(file_path):
    """
    Parses an XML file and extracts relevant information into a dictionary.
    """
    print('Importing metadata from the .xml file...')

    # account for file name variations
    try:
        tree = ET.parse(file_path)
    except FileNotFoundError:
        # Only modify the name part, not the whole path
        parent = file_path.parent
        name = file_path.name
        if '-' in name:
            alternative_name = name.replace('-', '_')
        else:
            alternative_name = name.replace('_', '-')

        alternative_path = parent / alternative_name
        tree = ET.parse(alternative_path)

    myroot = tree.getroot()

    data = {
        "nbits": int(myroot.find("acquisitionSystem").find("nBits").text),
        "dat_sampling_rate": None,
        "n_channels": None,
        "voltage_range": None,
        "amplification": None,
        "offset": None,
        "eeg_sampling_rate": None,
        "anatomical_groups": [],
        "skipped_channels": [],
        "discarded_channels": [],
        "spike_detection": [],
        "units": [],
        "spike_groups": []
    }

    for sf in myroot.findall("acquisitionSystem"):
        data["dat_sampling_rate"] = int(sf.find("samplingRate").text)
        data["n_channels"] = int(sf.find("nChannels").text)
        data["voltage_range"] = float(sf.find("voltageRange").text)
        data["amplification"] = float(sf.find("amplification").text)
        data["offset"] = float(sf.find("offset").text)

    for val in myroot.findall("fieldPotentials"):
        data["eeg_sampling_rate"] = int(val.find("lfpSamplingRate").text)

    anatomical_groups, skipped_channels = [], []
    for x in myroot.findall("anatomicalDescription"):
        for y in x.findall("channelGroups"):
            for z in y.findall("group"):
                chan_group = []
                for chan in z.findall("channel"):
                    if int(chan.attrib.get("skip", 0)) == 1:
                        skipped_channels.append(int(chan.text))
                    chan_group.append(int(chan.text))
                if chan_group:
                    anatomical_groups.append(np.array(chan_group))

    if data["n_channels"] is not None:
        data["discarded_channels"] = np.setdiff1d(
            np.arange(data["n_channels"]), np.concatenate(anatomical_groups) if anatomical_groups else []
        )

    data["anatomical_groups"] = np.array(anatomical_groups, dtype="object")
    data["skipped_channels"] = np.array(skipped_channels)

    # Parse spike detection groups
    spike_groups = []
    for x in myroot.findall("spikeDetection/channelGroups"):
        for y in x.findall("group"):
            chan_group = [int(chan.text) for chan in y.find("channels").findall("channel")]
            if chan_group:
                spike_groups.append(np.array(chan_group))

    data["spike_groups"] = spike_groups
    # data["spike_groups"] = np.array(spike_groups, dtype="object")

    return data
def get_ttl_from_analog(rec_path, channel: int):

    """
    Extract TTL pulse edges from an OpenEphys continuous DAT file for a specified channel.

    Parameters
    ----------
    rec_path : str or pathlib.Path
        Path to the recording folder containing 'continuous.dat' and 'structure.oebin'.
    channel : int
        Index of the channel to extract TTL pulses from (0-based).

    Returns
    -------
    rising_edges : np.ndarray
        Indices of samples where the TTL signal rises (0 → 1 transitions).
    falling_edges : np.ndarray
        Indices of samples where the TTL signal falls (1 → 0 transitions).
    total_samples : int
        Total number of samples in the selected channel of the continuous recording.

    Raises
    ------
    TypeError
        If `channel` is not an integer.
    FileNotFoundError
        If the recording folder, 'continuous.dat', or 'structure.oebin' cannot be found.

    Notes
    -----
    - The function normalizes the analog signal of the selected channel to the range [0,1],
      thresholds it at 0.5 to create a digital 0/1 signal, and then detects rising and falling edges.
    - This function relies on `nap.load_eeg` for loading the continuous data.
    - The returned edges are in sample indices relative to the start of the recording.
    """

    if not isinstance(channel, int):
        raise TypeError(f"channel must be an int, got {type(channel).__name__}")

    rec_path = Path(rec_path)  # convert to Path

    if not rec_path.exists():
        raise FileNotFoundError(f"recording folder not found: {rec_path}")

    dat_path = list(rec_path.rglob("continuous.dat"))[0]  # look for continuous.dat inside the path
    if not dat_path.exists():
        raise FileNotFoundError(f"continuous.dat file not found: {dat_path}")

    oebin_path = list(rec_path.rglob("structure.oebin"))[0]  # look for structure.oebin inside the path
    if not oebin_path.exists():
        raise FileNotFoundError(f"structure.oebin file (OpenEphys metadata) not found: {oebin_path}")

    # Get OpenEphys metadata from structure.oebin file
    with open(oebin_path, 'r') as f:
        oebin = json.load(f)

    sample_rate = oebin["continuous"][0]["sample_rate"]
    num_channels = oebin["continuous"][0]["num_channels"]

    # lazy load DAT file
    signal = nap.load_eeg(filepath=str(dat_path), channel=None, n_channels=num_channels,
                            frequency=float(sample_rate), precision='int16',
                            bytes_size=2)
    signal = signal[:, channel]  # get only probe channels
    total_samples = len(signal)  # total number of samples

    # detect TTL rise and fall times
    # Normalize to 0..1
    signal = signal - np.min(signal)
    signal = signal / np.max(signal)

    # Convert to binary (0/1) using midpoint threshold
    signal = signal > 0.5  # True = high, False = low

    # Detect edges
    diff = np.diff(signal.astype(int))
    rising_edges = np.where(diff == 1)[0] + 1
    falling_edges = np.where(diff == -1)[0] + 1

    return rising_edges, falling_edges, total_samples


def align_tracking(
        recfolder_path,
        ttl_channel:int,
        column_names,
        use_bonsai_timestamps=False
):
    """
    Align Bonsai tracking data with OpenEphys global clock using TTL pulses.

    Parameters
    ----------
    recfolder_path : Path
        Path to session folder containing recording* subfolders.
    ttl_channel : int
        TTL channel used for frame sync.
    column_names : list
        Names to assign to the Bonsai tracking columns. You must provide the following:
        'x_left','y_left','x_right','y_right'
    use_bonsai_timestamps : bool
        If True → use Bonsai timestamps shifted to recording start.
        If False → use DAT-derived TTL timestamps (recommended).

    Returns
    -------
    nap.TsdFrame
        Tracking data aligned to DAT global clock.
    """

    # ============================================================
    # Locate recording folders
    # ============================================================
    recfolder_path = Path(recfolder_path)
    node_path = next((p for p in recfolder_path.rglob("Record Node*") if p.is_dir()), None) # spikeinterface also creates a folder 'recording' so going deeper
    recording_folders = [p for p in node_path.rglob("recording*") if p.is_dir()]
    recording_folders.sort(key=lambda p: (len(p.parts), p.name))

    aligned_tracking = []
    samples_so_far = 0  # running count of samples across recordings

    for rec_folder in recording_folders:

        print(f"\nRecording path: {rec_folder}")

        # --------------------------------------------------------
        # Load OpenEphys metadata
        # --------------------------------------------------------
        oebin_path = rec_folder / "structure.oebin"
        if not oebin_path.exists():
            raise FileNotFoundError(f"Missing: {oebin_path}")

        with open(oebin_path, "r") as f:
            oebin = json.load(f)

        sample_rate = oebin["continuous"][0]["sample_rate"]

        # --------------------------------------------------------
        # Load TTL events (in sample units)
        # --------------------------------------------------------
        ttl_starts, ttl_ends, samples_rec = get_ttl_from_analog(rec_folder, ttl_channel)
        ttl_starts = ttl_starts + samples_so_far  # make timestamps global
        samples_so_far += samples_rec  # update counter for next recording

        ttl_timestamps = ttl_starts.astype(float) / sample_rate  # convert to seconds

        # --------------------------------------------------------
        # Load Bonsai CSV tracking
        # --------------------------------------------------------
        bonsai_files = list(rec_folder.glob("BonsaiTracking*.csv"))
        if len(bonsai_files) == 0:
            print("No Bonsai CSV found.")
            continue
        if len(bonsai_files) > 1:
            print("Warning: multiple Bonsai CSVs found. Skipping folder.")
            continue

        bonsai_file = pd.read_csv(bonsai_files[0], header=0, dtype=str)

        print(f"TTL pulses:         {len(ttl_timestamps)}")
        print(f"Bonsai frames:      {len(bonsai_file)}")

        # --------------------------------------------------------
        # Compute Bonsai timestamps in seconds
        # --------------------------------------------------------
        bonsai_timestamps = bonsai_file.iloc[:, 0].apply(parser.isoparse)
        bonsai_timestamps = (
                                    bonsai_timestamps - bonsai_timestamps.iloc[0]
                            ).dt.total_seconds() + ttl_timestamps[0]

        # --------------------------------------------------------
        # Trim if TTL pulses > Bonsai frames
        # --------------------------------------------------------
        extra = len(ttl_timestamps) - len(bonsai_file)
        if extra > 0:
            print(f"Trimming {extra} extra TTL pulses at the end")
            ttl_timestamps = ttl_timestamps[: len(bonsai_file)]
            if extra > 3:
                print("Warning: many missing frames!")
        elif extra < 0:
            # More Bonsai frames than TTL pulses
            print(f"Warning: {abs(extra)} extra Bonsai frames detected. Did you stop Bonsai too late? Trimming to match TTLs.")
            bonsai_file = bonsai_file.iloc[:len(ttl_timestamps)]
            bonsai_timestamps = bonsai_timestamps[:len(ttl_timestamps)]

        # --------------------------------------------------------
        # Assemble tracking DataFrame
        # --------------------------------------------------------
        tracking_numeric = bonsai_file.iloc[:, 1:1 + len(column_names)].astype(float)
        tracking_numeric.columns = column_names

        if use_bonsai_timestamps:
            timestamps = bonsai_timestamps.to_numpy()
        else:
            timestamps = ttl_timestamps

        df_track = pd.DataFrame(tracking_numeric)
        df_track.insert(0, "timestamps", timestamps)

        aligned_tracking.append(df_track)

    # ============================================================
    # Concatenate recordings
    # ============================================================
    aligned_tracking = pd.concat(aligned_tracking, ignore_index=True)
    aligned_tracking.set_index('timestamps', inplace=True)

    # ============================================================
    # Convert to Pynapple TsdFrame
    # ============================================================
    tracking_tsd = nap.TsdFrame(
        t=aligned_tracking,
        time_units="s"
    )

    print("\nTracking aligned.")
    return tracking_tsd

def process_led_tracking(
    tracking_data,
    pixel_width,
    min_spacing=3,
    max_spacing=8,
    plot=True
):
    """
    Clean dual-LED tracking data and compute:
        - position (x,y midpoint of LEDs)
        - head direction (0–2π)
        - velocity (cm/s)

    Automatically handles missing or bad frames:
        - If LED distance is outside min/max spacing → frame is NaN

    Parameters
    ----------
    aligned_tracking : pandas.DataFrame or nap.TsdFrame
        Must contain columns: ['x_left', 'y_left', 'x_right', 'y_right'].
        If TsdFrame, index is used as timestamps.

    pixel_width : float
        cm per pixel (calibration factor)

    min_spacing, max_spacing : float
        Allowed LED distance range in cm

    plot : bool
        If True, produce diagnostic plots.

    Returns
    -------
    dict
        Dictionary of Pynapple objects:
        {
            "pos": TsdFrame,
            "hd":  Tsd,
            "vel": Tsd
        }
    """

    # --- Convert TsdFrame to DataFrame if necessary ---
    if isinstance(tracking_data, nap.TsdFrame):
        tracking_data = tracking_data.as_dataframe()

    # --- Validate columns ---
    req = ['x_left', 'y_left', 'x_right', 'y_right']
    if not all(col in tracking_data.columns for col in req):
        raise ValueError(f"Bonsai tracking must contain columns: {req}")

    # --- Scale to pixel width ---
    tracking_data = tracking_data * pixel_width

    # --- Extract numeric arrays ---
    t = tracking_data.index.values
    xl = tracking_data['x_left'].to_numpy()
    yl = tracking_data['y_left'].to_numpy()
    xr = tracking_data['x_right'].to_numpy()
    yr = tracking_data['y_right'].to_numpy()

    # --- Compute position (midpoint) ---
    pos_x = (xl + xr) / 2
    pos_y = - (yl + yr) / 2 + np.nanmax((yl + yr) / 2)  # flip Y

    # --- LED spacing ---
    dx = xl - xr
    dy = yl - yr
    led_dist = np.hypot(dx, dy)

    # --- Head direction (0–2π) ---
    hd = np.mod(np.arctan2(dy, dx) - np.pi, 2 * np.pi)

    # --- Remove bad tracking ---
    mask = (led_dist >= min_spacing) & (led_dist <= max_spacing)
    if np.any(~mask):
        n_removed = np.sum(~mask)
        pct_removed = round(n_removed / len(mask) * 100, 1)
        print(f"{pct_removed}% frames removed due to LED spacing outside [{min_spacing},{max_spacing}] cm")

    pos_x[~mask] = np.nan
    pos_y[~mask] = np.nan
    hd[~mask] = np.nan

    # --- Velocity ---
    dt = np.median(np.diff(t))
    vel = np.hypot(np.diff(pos_x), np.diff(pos_y)) / dt
    vel = np.insert(vel, 0, np.nan)

    # --- Pynapple conversion ---
    tsd_pos = nap.TsdFrame(
        t=t,
        d=np.column_stack([pos_x, pos_y]),
        columns=['x', 'y'],
        time_units="s"
    )

    tsd_hd = nap.Tsd(t=t, d=hd, time_units="s")
    tsd_vel = nap.Tsd(t=t, d=vel, time_units="s")

    print("\nTracking processed.")

    # --- Diagnostic plots ---
    if plot:
        plt.figure()
        plt.hist(led_dist, bins=300, color='b')
        plt.axvline(min_spacing, color='r')
        plt.axvline(max_spacing, color='r')
        plt.title("LED spacing (thresholds in red)")
        plt.xlabel("Distance (cm)")
        plt.ylabel("# samples")

        plt.figure()
        plt.plot(pos_x, pos_y, 'k', linewidth=0.1)
        plt.title("Cleaned tracking")
        plt.xlabel("X (cm)")
        plt.ylabel("Y (cm)")
        plt.gca().set_aspect("equal", "box")

    return {"position": tsd_pos, "head-direction": tsd_hd, "velocity": tsd_vel}


### Export to Pynapple ###
def export_to_pynapple(
        sorting_analyzer_or_sorting: SortingAnalyzer | BaseSorting,
        attach_unit_metadata=True,
        segment_index=None,
):
    """
    This is a modification of to_pynapple_tsgroup function from SpikeInterface, with fixed bug with addding metadata
    Returns a pynapple TsGroup object based on spike train data.

    Parameters
    ----------
    sorting_analyzer_or_sorting : SortingAnalyzer
        A SortingAnalyzer object
    attach_unit_metadata : bool, default: True
        If True, any relevant available metadata is attached to the TsGroup. Will attach
        `unit_locations`, `quality_metrics` and `template_metrics` if computed. If False,
        no metadata is included.
    segment_index : int | None, default: None
        The segment index. Can be None if mono-segment sorting.

    Returns
    -------
    spike_train_TsGroup : pynapple.TsGroup
        A TsGroup object from the pynapple package.
    """

    if isinstance(sorting_analyzer_or_sorting, SortingAnalyzer):
        sorting = sorting_analyzer_or_sorting.sorting
    elif isinstance(sorting_analyzer_or_sorting, BaseSorting):
        sorting = sorting_analyzer_or_sorting
    else:
        raise TypeError(
            f"The `sorting_analyzer_or_sorting` argument must be a SortingAnalyzer or Sorting object, not a {type(sorting_analyzer_or_sorting)} type object."
        )

    unit_ids = sorting.unit_ids

    unit_ids_castable = True
    try:
        unit_ids_ints = [int(unit_id) for unit_id in unit_ids]
    except ValueError:
        warn_msg = "Pynapple requires integer unit ids, but `unit_ids` cannot be cast to int. "
        warn_msg += "We will set the index of the TsGroup to [0,1,2,...] and attach the original "
        warn_msg += "unit ids to the TsGroup as metadata with the name 'unit_id'."
        warn(warn_msg)
        unit_ids_ints = np.arange(len(unit_ids))
        unit_ids_castable = False

    spikes_trains = {
        unit_id_int: sorting.get_unit_spike_train(unit_id=unit_id, return_times=True, segment_index=segment_index)
        for unit_id_int, unit_id in zip(unit_ids_ints, unit_ids)
    }

    metadata_list = []
    if not unit_ids_castable:
        metadata_list.append(pd.DataFrame(unit_ids, columns=["unit_id"]))

    # Look for good metadata to add, if there is a sorting analyzer
    if attach_unit_metadata and isinstance(sorting_analyzer_or_sorting, SortingAnalyzer):

        metadata_list = []
        if (unit_locations := sorting_analyzer_or_sorting.get_extension("unit_locations")) is not None:
            array_of_unit_locations = unit_locations.get_data()
            n_dims = np.shape(sorting_analyzer_or_sorting.get_extension("unit_locations").get_data())[1]
            pd_of_unit_locations = pd.DataFrame(
                array_of_unit_locations, columns=["x", "y", "z"][:n_dims], index=unit_ids
            )
            metadata_list.append(pd_of_unit_locations)
        if (quality_metrics := sorting_analyzer_or_sorting.get_extension("quality_metrics")) is not None:
            metadata_list.append(quality_metrics.get_data())
        if (template_metrics := sorting_analyzer_or_sorting.get_extension("template_metrics")) is not None:
            metadata_list.append(template_metrics.get_data())

    if len(metadata_list) > 0:
        metadata = pd.concat(metadata_list, axis=1)
        metadata.index = unit_ids_ints
    else:
        metadata = None

    spike_train_tsgroup = nap.TsGroup(
        {unit_id: nap.Ts(spike_train) for unit_id, spike_train in spikes_trains.items()},
    )

    # Adrian's fix: add metadata dataframe as individual series:
    if metadata is not None:
        for col_name in metadata.columns:
            # Add each column (Series) to the TsGroup, using the column name as the key
            spike_train_tsgroup[col_name] = metadata[col_name]

    return spike_train_tsgroup



