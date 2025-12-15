import matplotlib.pyplot as plt
import pynapple as nap
import json
import pandas as pd
from dateutil import parser
import numpy as np
from pathlib import Path

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
    #dt = np.median(np.diff(t))
    #vel = np.hypot(np.diff(pos_x), np.diff(pos_y)) / dt
    #vel = np.insert(vel, 0, np.nan)

    # --- Pandas conversion ---
    tracking_df = pd.DataFrame(
        {
            "x": pos_x,
            "y": pos_y,
            "hd": hd
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