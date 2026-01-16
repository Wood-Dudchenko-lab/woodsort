import json
import pandas as pd
from pathlib import Path
import spikeinterface.full as si
import woodsort.neuroscope as neuroscope
import numpy as np
from probeinterface import write_probeinterface

def process_with_neuroscope(
    recording,
    probe_or_probegroup,
    session_path,
    shank_groups,
    save_path=None,
    plot_probe=False,
    plot_range=None,
    save_mapping=True,
    remove_bad_channels=True,
    excluded_channels=None
):
    """
    Prepare an OpenEphys SpikeInterface `Recording` using Neuroscope channel mapping.

    This function loads the Neuroscope channel mapping from `session_path`, uses
    `shank_groups` to select the relevant channels, writes that mapping onto the
    provided probe, attaches the probe to the recording grouped by shank, optionally
    plots probe maps and trace snippets, optionally removes user-specified channels,
    concatenates/splits the recording by shank groups, applies a high-pass/bandpass
    filter, optionally detects and removes bad channels, and optionally saves a
    `ChannelMapping.csv` that includes a boolean `is_bad` column.

    Parameters
    ----------
    recording : spikeinterface.core.BaseRecording
        The raw OpenEphys recording (or any SpikeInterface-compatible recording).
    probe_or_probegroup : probeinterface.Probe or probeinterface.ProbeGroup
        Probe definition to attach to the recording.
    session_path : pathlib.Path or str
        Path to the folder containing the openephys session directory
    shank_groups : list[int]
        Channel group IDs (as defined by the Neuroscope mapping) to include.
        Only rows whose `channel_group` is in this list are used to build the
        probe-to-recording mapping.
    save_path : Path, default None
        Path to the folder where files should be saved
    plot_probe : bool, default False
        If True, plot probe maps with device indices and with channel IDs.
    plot_range : tuple[float, float] or list[float, float] or None, default None
        If provided, must be a 2-element (t_start, t_end) time range (in seconds)
        used to plot traces for each shank-group recording after splitting.
    save_mapping : bool, default True
        If True, write `ChannelMapping.csv` to `session_path`. The saved table
        is the Neuroscope channel mapping plus an `is_bad` column when
        `remove_bad_channels=True`.
    remove_bad_channels : bool, default True
        If True, run `si.detect_and_remove_bad_channels(...)` on the split
        recordings and mark the detected channels in the saved channel mapping.
    excluded_channels : list[str] or None, default None
        Optional list of channel identifiers to remove *before* concatenation/splitting.
        Accepted formats are:
          - "CH17" style strings (must be "CH" + digits), with surrounding whitespace allowed
          - Bare digit strings like "17" (auto-converted to "CH17")
        Any other format raises a ValueError/TypeError.

    Returns
    -------
    dict
        A dictionary of split recordings keyed by group/shank id (i.e. the output
        of `recording.split_by("group")`), after optional channel exclusion,
        filtering, and optional bad-channel removal.

    Raises
    ------
    TypeError
        If `excluded_channels` is not a list (when not None), or if any entry in
        `excluded_channels` is not a string.
    ValueError
        If `plot_range` is provided but is not a 2-element list/tuple, or if any
        `excluded_channels` entry is not in an accepted format.

    Notes
    -----
    - The mapping saved to CSV is derived from Neuroscope’s channel mapping and
      is annotated with `is_bad` using 0-based channel indices extracted from
      SpikeInterface bad-channel IDs.
    - `excluded_channels` removal happens on the recording *after* setting the
      probe, but *before* concatenation/splitting.
    """

    session_path = Path(session_path)
    if save_path is None:
        save_path = session_path
    else:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    # Validate plot_range
    if plot_range is not None:
        if not isinstance(plot_range, (list, tuple)) or len(plot_range) != 2:
            raise ValueError(
                "plot_range must be None or a list/tuple of length 2 "
                "(t_start, t_end)"
            )

    # Change channel ids in the recording to match Neuroscope (0-base sequential integers)
    channel_ids = recording.get_channel_ids()
    new_channel_ids = [int(ch[2:]) - 1 for ch in channel_ids]
    recording = recording.rename_channels(new_channel_ids)

    # Load Neuroscope channel mapping
    channel_mapping = neuroscope.load_neuroscope_channels(session_path)

    # Get channel indices corresponding to the probe
    neuroscope_channel_indices = (
        channel_mapping.loc[
            channel_mapping["channel_group"].isin(shank_groups),
            "channel_0based",
        ].tolist()
    )


    # Get probe specs to re-add later (SpikeIntrface bug)
    #manufacturer = probe.manufacturer
    #model_name = probe.model_name

    # Add neuroscope mapping to probe
    probe_or_probegroup = neuroscope.add_neuroscope_mapping(
        probe_or_probegroup,
        neuroscope_channel_indices
    )

    # Save probe to file
    write_probeinterface(save_path / 'probegroup.json', probe_or_probegroup)

    # Attach probe
    recording = recording.set_probe(probe_or_probegroup, group_mode="by_shank")

    if plot_probe:
        si.plot_probe_map(recording, with_device_index=True)
        si.plot_probe_map(recording, with_channel_ids=True)

    # Exclude channels if needed:
    if excluded_channels is not None:
        if not isinstance(excluded_channels, list):
            raise TypeError("excluded_channels must be a list or None")

        if not all(isinstance(ch, int) for ch in excluded_channels):
            raise TypeError("All excluded_channels must be integers")

        recording = recording.remove_channels(excluded_channels)
        print(f"Excluded channels (0-based Neuroscope index): {excluded_channels}")

    # now concatenate all recordings and split by shank
    recording = si.concatenate_recordings([recording])
    recording = recording.split_by("group")

    for group_id, rec_g in recording.items():
        if plot_range is not None:
            si.plot_traces(
                rec_g,
                order_channel_by_depth=True,
                time_range=list(plot_range)
            )

    # filter
    print('Applying high-pass filter (300-6000 Hz)...')
    recording = si.bandpass_filter(recording)

    # Remove bad channels and print
    bad_ids = []
    if remove_bad_channels:
        print('Detecting and removing bad channels...')
        recording = si.detect_and_remove_bad_channels(recording)

        # get IDs of bad channels
        for rec in recording.values():
            bad = rec._kwargs.get("bad_channel_ids")
            if bad is not None:
                bad_ids.extend(bad)

        bad_ids = [int(x) for x in bad_ids]
        print(f'Bad channel IDs (0-based Neuroscope index): {bad_ids}')

    # start by assuming all channels are good
    channel_mapping['is_good'] = True

    # mark bad channels
    channel_mapping.loc[
        channel_mapping["channel_0based"].isin(bad_ids),
        "is_good"
    ] = False

    # mark channels not in shank_groups as bad
    channel_mapping.loc[
        ~channel_mapping["channel_group"].isin(shank_groups),
        "is_good"
    ] = False

    # save channel mapping
    if save_mapping:
        out_path = save_path / "ChannelMapping.csv"
        channel_mapping.to_csv(out_path, index=True)
        print(f'Channel mapping saved to {out_path} \n')

    return recording

def get_events(rec_path, epochs='all', save_output=True, save_path=None,
               save_name="EventTimestamps.csv"):
    """
    Extract TTL event timestamps from OpenEphys recordings across one or more epochs.

    Parameters
    ----------
    rec_path : str or Path
        Path to the folder containing the OpenEphys recording
    epochs : 'all' or list of int, optional
        Epoch indices to extract events from. If 'all' (default), all epochs are used.
    save_output : bool, optional
        If True (default), the resulting DataFrame is saved as a CSV file.
    save_path : str or Path, optional
        Directory to save the CSV file. If None (default), it will be saved in `recfolder_path`.
    save_name : str, optional
        Name of the output CSV file (default is "EventTimestamps.csv"). '.csv' extension will
        be added if not provided.

    Returns
    -------
    events_all : pandas.DataFrame
        DataFrame containing all extracted events with columns:
        - 'time': timestamp in seconds
        - 'event': event state (1 for TTL on, -1 for TTL off)

    """

    rec_path = Path(rec_path)
    if save_path is None:
        save_path = rec_path
    else:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    # Get epoch timestamps
    epoch_df = get_epochs(rec_path, verbose=False, save_output=False, as_samples=True)
    epoch_starts = epoch_df['Start'].values
    epoch_ends = epoch_df['End'].values

    # Get epoch indices and run checks
    n_epochs_total = len(epoch_starts)
    if epochs == 'all':
        epochs_to_use = set(range(n_epochs_total))
    else:
        epochs_to_use = set(epochs)  # make sure it's a list-like of ints

        if not all(isinstance(e, int) for e in epochs_to_use):
            raise TypeError("epochs must be 'all' or a list of integers")

        invalid = [e for e in epochs_to_use if e < 0 or e >= n_epochs_total]
        if invalid:
            raise ValueError(f"Invalid epoch indices: {invalid}")

    # now get event timestamps

    # Locate the Record Node folder
    node_path = next(
        (p for p in rec_path.rglob("Record Node*") if p.is_dir()),
        None
    )
    if node_path is None:
        raise FileNotFoundError("No 'Record Node*' folder found under this path.")

    # Collect recording folders
    recording_folders = [p for p in node_path.rglob("recording*") if p.is_dir()]
    recording_folders.sort(key=lambda p: (len(p.parts), p.name))

    if len(recording_folders) == 0:
        raise FileNotFoundError("No 'recording*' folders found under the Record Node.")

    # Loop through recordings and extract event times
    events_all = pd.DataFrame()
    for n_rec, rec_folder in enumerate(recording_folders):

        print(f"Extracting event times: {rec_folder}")

        # skip epochs not on the list
        if n_rec not in epochs_to_use:
            continue

        # Load sample rate from structure.oebin
        oebin_path = rec_folder / "structure.oebin"
        if not oebin_path.exists():
            raise FileNotFoundError(f"Missing: {oebin_path}")

        with open(oebin_path, "r") as f:
            oebin = json.load(f)

        sample_rate = oebin["continuous"][0]["sample_rate"]

        # locate sync_messages.txt to get the 'rec' start timestamp in 'play' time
        sync_path = rec_folder / "sync_messages.txt"
        if not sync_path.exists():
            raise FileNotFoundError(f"Missing: {sync_path}")

        with open(sync_path, "r") as f:
            last_line = f.readlines()[-1]

        rec_start = int(last_line.split()[-1])

        # load digital inputs
        states = np.load(rec_folder / 'events' / 'Acquisition_Board-100.Rhythm Data' / 'TTL' / 'states.npy')
        sample_numbers = np.load(
            rec_folder / 'events' / 'Acquisition_Board-100.Rhythm Data' / 'TTL' / 'sample_numbers.npy')
        sample_numbers -= rec_start  # count from the beginning of the recording
        sample_numbers += epoch_starts[n_rec]  # offset by recording start
        timestamps = sample_numbers / sample_rate  # convert to seconds

        # convert to dataframe
        events = pd.Series(states, index=timestamps)
        events = events.reset_index()
        events.columns = ['time', 'event']

        events_all = pd.concat([events_all, events], ignore_index=True)

        if len(events_all) == 0:
            print(f"\nNo event timestamps detected in: {rec_path}\n")
            return

        if save_output:
            if save_path is not None:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
            else:
                save_path = rec_path

            if not str(save_name).lower().endswith(".csv"):
                save_name += ".csv"

            outfile = save_path / save_name
            events_all.to_csv(outfile, index=False)
            print(f"\nEvent timestamps saved to: {outfile}\n")

    return events_all


def get_epochs(rec_path, save_path=None, save_name='EpochTimestamps.csv', verbose=True, save_output=True, as_samples=False):

    """
    Extract start and end timestamps for each recording epoch in an OpenEphys session.

    Each 'recording*' folder within a 'Record Node*' directory must contain exactly one
    'sample_numbers.npy' file, which is used to determine the epoch boundaries.

    Parameters
    ----------
    rec_path : str or Path
        Path to the session folder containing 'Record Node*/recording*' folders.

    save_path : str or Path, optional
        Directory to save the CSV file. If it does not exist, it will be created.
        If None (default), the CSV is saved in `recfolder_path`.

    save_name : str, optional
        Name of the output CSV file (default: 'EpochTimestamps.csv'). The '.csv'
        extension will be added if not provided.

    verbose : bool, optional
        If True (default), prints progress messages during extraction.

    save_output : bool, optional
        If True (default), saves the resulting DataFrame to CSV.

    as_samples : bool, optional
        If True, the returned epoch start/end times are in raw sample numbers rather than seconds.

    Returns
    -------
    epoch_df : pandas.DataFrame
        DataFrame containing all epoch start and end times. Columns:
        - 'Start' : epoch start time in seconds (or samples if `as_samples=True`)
        - 'End'   : epoch end time in seconds (or samples if `as_samples=True`)
    if verbose:
        print('\nExtracting recording boundaries...')

    """

    rec_path = Path(rec_path)

    # ------------------------------------------------------------
    # Locate the Record Node folder
    # ------------------------------------------------------------
    node_path = next(
        (p for p in rec_path.rglob("Record Node*") if p.is_dir()),
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
        if verbose:
            print(f"Extracting start and end times: {rec_folder}")

        # Load sample rate from structure.oebin
        oebin_path = rec_folder / "structure.oebin"
        if not oebin_path.exists():
            raise FileNotFoundError(f"Missing: {oebin_path}")

        with open(oebin_path, "r") as f:
            oebin = json.load(f)

        sample_rate = oebin["continuous"][0]["sample_rate"]

        # --------------------------------------------------------
        # locate sync_messages.txt to get the 'rec' start timestamp in 'play' time
        # --------------------------------------------------------
        sync_path = rec_folder / "sync_messages.txt"
        if not sync_path.exists():
            raise FileNotFoundError(f"Missing: {sync_path}")

        with open(sync_path, "r") as f:
            last_line = f.readlines()[-1]

        rec_start = int(last_line.split()[-1])

        # --------------------------------------------------------
        # Locate sample_numbers.npy to get start and end timestamps in 'play' time
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
        # Convert start/end to *global* sample numbers (all recordings merged together)
        # --------------------------------------------------------
        start_sample = sample_numbers[0] - rec_start + samples_so_far
        end_sample = sample_numbers[-1] - rec_start + samples_so_far

        # Store epoch (in seconds or samples)
        if as_samples:
            epochs_all.append([start_sample, end_sample])
        else:
            epochs_all.append([start_sample / sample_rate, end_sample / sample_rate])

        # Update running offset for next recording
        samples_so_far += (end_sample - start_sample) + 1

    # ------------------------------------------------------------
    # Build DataFrame
    # ------------------------------------------------------------
    epoch_df = pd.DataFrame(epochs_all, columns=["Start", "End"])

    if save_output:
        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = rec_path

        if not str(save_name).lower().endswith(".csv"):
            save_name += ".csv"

        outfile = save_path / save_name
        epoch_df.to_csv(outfile, index=False)
        print(f"\nEpoch timestamps saved to: {outfile}\n")

    return epoch_df


def get_openephys_metadata(rec_path, save_path=None, save_name="MetadataOpenephys.txt"):
    """
    Extract and save Open Ephys sync metadata from a recording session.

    This function searches an Open Ephys session directory for
    'Record Node*/recording*' folders, reads the contents of
    'sync_messages.txt' from each recording folder, removes line breaks,
    and writes the resulting messages to a single text file with one
    message per line.

    The order of messages in the output file follows the chronological
    order of the recording folders.

    Parameters
    ----------
    rec_path : str or Path
        Path to the Open Ephys session directory containing
        'Record Node*' folders.

    save_path : str or Path, optional
        Directory to save the metadata file. Created if it does not exist.

    save_name : str, optional
        Name of the output text file written to `recfolder_path`
        (default: "MetadataOpenephys.txt").

    Returns
    -------
    messages : list of str
        A list of sync message strings, one per recording folder,
        with newline characters removed.

    Raises
    ------
    FileNotFoundError
        If no 'Record Node*' folder, no 'recording*' folders, or a
        required 'sync_messages.txt' file is missing.
    """

    print("\nCollecting sync messages...")
    rec_path = Path(rec_path)

    node_path = next(
        (p for p in rec_path.rglob("Record Node*") if p.is_dir()),
        None
    )
    if node_path is None:
        raise FileNotFoundError("No 'Record Node*' folder found under this path.")

    recording_folders = [p for p in node_path.rglob("recording*") if p.is_dir()]
    recording_folders.sort(key=lambda p: (len(p.parts), p.name))

    if not recording_folders:
        raise FileNotFoundError("No 'recording*' folders found under the Record Node.")

    messages = []

    for rec_folder in recording_folders:
        sync_path = rec_folder / "sync_messages.txt"
        if not sync_path.exists():
            raise FileNotFoundError(f"Missing: {sync_path}")

        text = sync_path.read_text(encoding="utf-8", errors="replace")
        text = " ".join(text.splitlines())  # remove \n cleanly
        messages.append(text)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = rec_path

    with open(save_path / save_name, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(f"{msg}\n")

    print(f"Sync metadata written to: {save_path / save_name}\n")

    return messages






