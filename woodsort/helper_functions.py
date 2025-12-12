
from __future__ import annotations
import matplotlib.pyplot as plt
import pynapple as nap
import json
import pandas as pd
from dateutil import parser
from warnings import warn
from spikeinterface.core import SortingAnalyzer, BaseSorting
import re
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import os
from scipy.signal import firwin, lfilter, resample_poly
from tqdm import tqdm

def get_lfp_from_dat(basepath, outFs=1250, lopass=450, noPrompts=True):
    # Determine basename
    basename = os.path.basename(os.path.normpath(basepath))
    dat_file = os.path.join(basepath, f"{basename}.dat")
    lfp_file = os.path.join(basepath, f"{basename}.lfp")
    xml_file = os.path.join(basepath, f"{basename}.xml")

    # If dat file doesn't exist, try amplifier.dat
    if not os.path.exists(dat_file):
        dat_file = os.path.join(basepath, "amplifier.dat")
        if not os.path.exists(dat_file):
            raise FileNotFoundError("Dat file does not exist")

    # Read metadata from XML if available
    inFs = 30000  # default sampling frequency
    nbChan = 16   # default number of channels
    if os.path.exists(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        inFs = float(root.findtext('SampleRate', default=inFs))
        nbChan = int(root.findtext('nChannels', default=nbChan))
        outFs = float(root.findtext('lfpSampleRate', default=outFs))

    # Calculate filter and downsample parameters
    if lopass > outFs / 2:
        raise ValueError("Low-pass frequency exceeds Nyquist limit")
    downsample_ratio = int(inFs / outFs)
    nyquist = inFs / 2
    norm_cutoff = lopass / nyquist

    # FIR filter (sinc)
    ntbuff = 525  # filter length
    fir_coeff = firwin(ntbuff, norm_cutoff)

    # Check file size
    size_in_bytes = 2  # int16
    file_size = os.path.getsize(dat_file)
    chunk_size = int(1e5)
    if chunk_size % downsample_ratio != 0:
        chunk_size += downsample_ratio - (chunk_size % downsample_ratio)
    nbChunks = file_size // (nbChan * size_in_bytes * chunk_size)

    # Open files
    with open(dat_file, 'rb') as fdat, open(lfp_file, 'wb') as flfp:
        for ibatch in tqdm(range(nbChunks), desc="Processing LFP"):
            if ibatch > 0:
                # Move back ntbuff samples for filter overlap
                fdat.seek((ibatch * chunk_size - ntbuff) * nbChan * size_in_bytes)
                dat = np.fromfile(fdat, dtype=np.int16, count=nbChan*(chunk_size+2*ntbuff))
                dat = dat.reshape(nbChan, chunk_size + 2*ntbuff)
            else:
                dat = np.fromfile(fdat, dtype=np.int16, count=nbChan*(chunk_size + ntbuff))
                dat = dat.reshape(nbChan, chunk_size + ntbuff)

            lfp_data = np.zeros((nbChan, chunk_size // downsample_ratio), dtype=np.int16)
            for ch in range(nbChan):
                filtered = lfilter(fir_coeff, 1.0, dat[ch, :])
                # Downsample
                lfp_data[ch, :] = filtered[ntbuff::downsample_ratio][:chunk_size // downsample_ratio].astype(np.int16)

            flfp.write(lfp_data.T.tobytes())

        # Handle remainder
        remainder = file_size // (size_in_bytes * nbChan) - nbChunks*chunk_size
        if remainder > 0:
            fdat.seek((nbChunks*chunk_size - ntbuff) * nbChan * size_in_bytes)
            dat = np.fromfile(fdat, dtype=np.int16, count=nbChan*(remainder + ntbuff))
            dat = dat.reshape(nbChan, remainder + ntbuff)
            lfp_data = np.zeros((nbChan, remainder // downsample_ratio), dtype=np.int16)
            for ch in range(nbChan):
                filtered = lfilter(fir_coeff, 1.0, dat[ch, :])
                lfp_data[ch, :] = filtered[ntbuff::downsample_ratio][:remainder // downsample_ratio].astype(np.int16)
            flfp.write(lfp_data.T.tobytes())

    print(f"{basename}.lfp file created!")

# Example usage:
# process_lfp_from_dat("/path/to/basepath", outFs=1250, lopass=450)

def get_epochs_openephys(recfolder_path, save_path=None, save_name='EpochTimestamps.csv'):
    """
    Extracts epoch start/end timestamps (in seconds) from OpenEphys recordings.
    Each 'recording*' folder must contain exactly one 'sample_numbers.npy' file.

    Parameters
    ----------
    recfolder_path : str or Path
        Path to the session folder containing 'Record Node*/recording*' folders.

    save_path : str or Path, optional
        Directory to save the epoch CSV. Created if it does not exist.

    save_name : str, optional
        Output CSV filename (default: 'EpochTimestamps.csv').

    Returns
    -------
    epoch_df : pd.DataFrame
        Columns:
            'start' : epoch start time in seconds
            'end'   : epoch end time in seconds
    """
    print('\nExtracting epoch timestamps...')
    recfolder_path = Path(recfolder_path)

    # ------------------------------------------------------------
    # Locate the Record Node folder
    # ------------------------------------------------------------
    node_path = next(
        (p for p in recfolder_path.rglob("Record Node*") if p.is_dir()),
        None
    )
    if node_path is None:
        raise FileNotFoundError("No 'Record Node*' folder found under this path.")

    # Collect recording folders
    recording_folders = [p for p in node_path.rglob("recording*") if p.is_dir()]
    recording_folders.sort(key=lambda p: (len(p.parts), p.name))

    if len(recording_folders) == 0:
        raise FileNotFoundError("No 'recording*' folders found under the Record Node.")

    # ------------------------------------------------------------
    # Loop through recordings and extract epoch times
    # ------------------------------------------------------------
    epochs_all = []
    samples_so_far = 0

    for rec_folder in recording_folders:

        print(f"\nExtracting start and end times: {rec_folder}")

        # Load sample rate from structure.oebin
        oebin_path = rec_folder / "structure.oebin"
        if not oebin_path.exists():
            raise FileNotFoundError(f"Missing: {oebin_path}")

        with open(oebin_path, "r") as f:
            oebin = json.load(f)

        sample_rate = oebin["continuous"][0]["sample_rate"]

        # --------------------------------------------------------
        # Locate sample_numbers.npy
        # --------------------------------------------------------
        continuous_folder = rec_folder / "continuous"
        sample_files = list(continuous_folder.rglob("sample_numbers.npy"))

        if len(sample_files) == 0:
            raise FileNotFoundError(f"No sample_numbers.npy found under: {rec_folder}")

        if len(sample_files) > 1:
            raise RuntimeError(
                f"Multiple sample_numbers.npy files found under {rec_folder}. "
                "There must be exactly one."
            )

        sample_file = sample_files[0]

        # Read the sample numbers
        sample_numbers = np.load(sample_file)

        if sample_numbers.ndim != 1 or sample_numbers.size < 2:
            raise ValueError(f"sample_numbers.npy in {rec_folder} is malformed.")

        # --------------------------------------------------------
        # Convert start/end to *global* sample numbers
        # --------------------------------------------------------
        start_sample = sample_numbers[0] + samples_so_far
        end_sample = sample_numbers[-1] + samples_so_far

        # Store epoch (in seconds)
        epochs_all.append([
            start_sample / sample_rate,
            end_sample / sample_rate
        ])

        # Update running offset for next recording
        samples_so_far += (end_sample - start_sample)

    # ------------------------------------------------------------
    # Build DataFrame
    # ------------------------------------------------------------
    epoch_df = pd.DataFrame(epochs_all, columns=["start", "end"])

    # ------------------------------------------------------------
    # Save if requested
    # ------------------------------------------------------------
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = recfolder_path

    if not str(save_name).lower().endswith(".csv"):
        save_name += ".csv"

    outfile = save_path / save_name
    epoch_df.to_csv(outfile, index=False)
    print(f"\nEpoch timestamps saved to: {outfile}\n")

    return epoch_df

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


def load_neuroscope_channels(recfolder_path, shank_groups=None):
    """
    Locate the session's continuous.xml file, extract anatomical channel groups,
    and return a 1D array of channel indices for the specified shank groups.

    Parameters
    ----------
    recfolder_path : str or Path
        Path to the session folder containing recording* subfolders.
    shank_groups : list, array, or None
        Indices of anatomical groups to extract.
        Must be specified; if None, an error is raised.

    Returns
    -------
    xml_channel_indices : np.ndarray
        Flattened array of channel indices belonging to the selected shanks.
    """

    if shank_groups is None:
        raise ValueError(
            "shank_groups must be specified (e.g. [0], [0,1], etc.) "
            "because this function only returns channel indices."
        )

    recfolder_path = Path(recfolder_path)

    # ------------------------------------------------------------
    # Find XML file
    # ------------------------------------------------------------
    xml_candidates = list(recfolder_path.rglob("continuous.xml"))
    if len(xml_candidates) == 0:
        raise FileNotFoundError("No continuous.xml found under this recording folder.")

    # Select the XML with smallest recording number
    xml_path = sorted(
        xml_candidates,
        key=lambda x: int(re.search(r"recording(\\d+)", str(x)).group(1))
    )[0]

    print(f"Using XML file: {xml_path}")

    # ------------------------------------------------------------
    # Parse XML (fallback name swap for hyphen/underscore)
    # ------------------------------------------------------------
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        alt_name = (xml_path.name.replace("-", "_")
                    if "-" in xml_path.name
                    else xml_path.name.replace("_", "-"))
        alt_path = xml_path.parent / alt_name
        print(f"Trying alternative XML filename: {alt_path}")
        tree = ET.parse(alt_path)

    root = tree.getroot()

    # ------------------------------------------------------------
    # Extract anatomical channel groups ONLY
    # ------------------------------------------------------------
    anatomical_groups = []

    for desc in root.findall("anatomicalDescription"):
        for cg in desc.findall("channelGroups"):
            for group in cg.findall("group"):
                channels = [int(ch.text) for ch in group.findall("channel")]
                if channels:
                    anatomical_groups.append(np.array(channels))

    if len(anatomical_groups) == 0:
        raise ValueError("No anatomical channel groups found in XML.")

    anatomical_groups = np.array(anatomical_groups, dtype=object)

    # ------------------------------------------------------------
    # Select shanks
    # ------------------------------------------------------------
    try:
        xml_channel_indices = np.concatenate(anatomical_groups[shank_groups])
    except IndexError:
        raise IndexError(
            f"shank_groups {shank_groups} exceed available groups "
            f"(found {len(anatomical_groups)})"
        )

    return xml_channel_indices


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


def align_tracking_bonsai(
        recfolder_path,
        ttl_channel:int,
        column_names,
        use_bonsai_timestamps=False,
        save_path=None,
        save_name='TrackingAligned.csv'
):
    """
    Align Bonsai tracking data with the OpenEphys global clock using TTL pulses.

    This function loads tracking data from each recording subfolder, aligns
    each frame to corresponding TTL timestamps, trims mismatches, and outputs
    a continuous tracking dataframe indexed by global time.

    Parameters
    ----------
    recfolder_path : Path or str
        Path to the session folder containing one or more 'recording*' subfolders.

    ttl_channel : int
        TTL channel used for frame synchronization between Bonsai and OpenEphys.

    column_names : list of str
        Names to assign to the Bonsai tracking columns.
        Must include (in order):
            'x_left', 'y_left', 'x_right', 'y_right'

    use_bonsai_timestamps : bool, default False
        If True:
            Use Bonsai timestamps shifted so that the first frame aligns with
            the first TTL pulse.
        If False (recommended):
            Use OpenEphys-derived TTL timestamps for each video frame.

    save_path : str or Path or None, default None
        Optional directory where the aligned tracking CSV will be saved.
        - If None → no file is saved.
        - If the directory does not exist, it is created automatically.
        - Must be a valid directory path (not a file).

    save_name : str, default 'TrackingAligned.csv'
        Filename for saving the aligned tracking table.
        - If it does not end with '.csv', the extension is appended automatically.

    Returns
    -------
    pandas.DataFrame
        A dataframe indexed by global timestamps (seconds), containing the
        aligned tracking coordinates. Columns include:

            ['x_left', 'y_left', 'x_right', 'y_right']

        The returned dataframe spans all recording subfolders concatenated
        in temporal order.
    """

    print("\nAligning Bonsai tracking data...")
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

    if save_path is not None:

        save_path = Path(save_path)

        # Create the directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)

        # Ensure .csv extension
        if not str(save_name).lower().endswith(".csv"):
            save_name = save_name + ".csv"

        # Save file
        aligned_tracking.to_csv(save_path / save_name, index=True)
        print(f"Tracking saved to: {save_path / save_name}")


    print("\nTracking aligned.\n")

    return aligned_tracking

def process_tracking_bonsai(
    tracking_data,
    pixel_width,
    min_spacing=3,
    max_spacing=8,
    plot=True,
    save_path=None,
    save_name='TrackingProcessed.csv'
):
    """
    Clean dual-LED tracking data and compute:
        - position (x, y midpoint of LEDs)
        - head direction (0–2π)
        - velocity (cm/s)

    Automatically removes bad frames where LED spacing is outside
    [min_spacing, max_spacing].

    Parameters
    ----------
    tracking_data : pandas.DataFrame or nap.TsdFrame
        Must contain columns: ['x_left', 'y_left', 'x_right', 'y_right'].
        If TsdFrame, its index is treated as timestamps.

    pixel_width : float
        Conversion factor from pixels → cm.

    min_spacing, max_spacing : float
        Allowed LED spacing range in centimeters.

    plot : bool
        If True, produce diagnostic plots.

    save_path : str or Path or None
        Directory where the processed tracking DataFrame will be saved.
        - If None: tracking is not saved.
        - If directory does not exist, it is automatically created.

    save_name : str
        Filename for saving the tracking CSV.
        - Must end with ".csv"; if not, ".csv" is added automatically.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by time with columns:
        ['x', 'y', 'hd', 'velocity']
    """

    print("\nProcessing aligned tracking")

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
    pos_y = -(yl + yr) / 2 + np.nanmax((yl + yr) / 2)  # flip Y

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
        print(f"{pct_removed}% frames removed due to LED spacing outside [{min_spacing}, {max_spacing}] cm")

    pos_x[~mask] = np.nan
    pos_y[~mask] = np.nan
    hd[~mask] = np.nan

    # --- Velocity ---
    dt = np.median(np.diff(t))
    vel = np.hypot(np.diff(pos_x), np.diff(pos_y)) / dt
    vel = np.insert(vel, 0, np.nan)

    # --- Pandas conversion ---
    tracking_df = pd.DataFrame(
        {
            "x": pos_x,
            "y": pos_y,
            "hd": hd,
            "velocity": vel
        },
        index=t
    )
    tracking_df.index.name = "timestamps"

    # ----------------------------------------------------
    # SAVE LOGIC — now auto-creates directory
    # ----------------------------------------------------
    if save_path is not None:

        save_path = Path(save_path)

        # Create the directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)

        # Ensure .csv extension
        if not str(save_name).lower().endswith(".csv"):
            save_name = save_name + ".csv"

        # Save file
        tracking_df.to_csv(save_path / save_name, index=True)
        print(f"Tracking saved to: {save_path / save_name}")

    print("\nTracking processed.\n")

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

    return tracking_df



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



