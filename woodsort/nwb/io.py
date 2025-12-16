import h5py
from scipy import io as spio
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import json
import re
import pynapple as nap
import pprint
from datetime import datetime
import pytz
from pathlib import Path


def get_openephys_events(datapath, foldername, time_offset=0, skip_first=0):
    # to do: account for instances when TTL is up at epoch edges

    print('Importing events from OpenEphys npy files...')
    # importing events
    states = np.load(datapath / foldername / 'Analysis' / 'states.npy')
    timestamps = np.load(datapath / foldername / 'Analysis' / 'timestamps.npy')
    events = pd.Series(states, index=timestamps + time_offset)
    events = events.iloc[skip_first:]

    df = events.reset_index()
    df.columns = ['time', 'event']

    # Identify start and end events
    df['event_type'] = df['event'].abs()
    df['event_sign'] = df['event'].apply(lambda x: 'start' if x > 0 else 'end')

    # Separate dataframes for each event type
    event_dict = {}
    for event_type in df['event_type'].unique():
        event_subset = df[df['event_type'] == event_type].copy()

        starts = event_subset[event_subset['event_sign'] == 'start'].reset_index(drop=True)
        ends = event_subset[event_subset['event_sign'] == 'end'].reset_index(drop=True)
        event_int = nap.IntervalSet(start=starts['time'].values, end=ends['time'].values)

        event_dict[event_type] = event_int

    return event_dict


def get_start_time(session_path):
    """
    Retrieve recording start time for a session folder.

    Priority order:
    1) If the folder contains Open Ephys 'Record Node*/recording*' folders,
       read 'sync_messages.txt' from the first recording folder.
    2) Otherwise, search recursively for exactly one metadata file:
       - MetadataOpenephys.txt (preferred)
       - Metadata.txt
    3) Otherwise, parse date from folder name.
    4) Final fallback: 2000-01-01 00:00:00 Europe/London.

    Raises an error if multiple metadata files of the same type are found.
    """

    tz = pytz.timezone("Europe/London")
    session_path = Path(session_path)

    # ------------------------------------------------------------
    # 1) Full Open Ephys session → parse sync_messages.txt
    # ------------------------------------------------------------
    node_path = next(
        (p for p in session_path.rglob("Record Node*") if p.is_dir()),
        None,
    )

    if node_path is not None:
        recording_folders = [
            p for p in node_path.rglob("recording*") if p.is_dir()
        ]
        recording_folders.sort(key=lambda p: (len(p.parts), p.name))

        if recording_folders:
            first_rec = recording_folders[0]
            sync_path = first_rec / "sync_messages.txt"

            if not sync_path.exists():
                raise FileNotFoundError(f"Missing: {sync_path}")

            text = sync_path.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"Software Time.*?:\s*(\d+)", text)

            if not m:
                raise ValueError(
                    f"\nNo 'Software Time' timestamp found in {sync_path}"
                )

            timestamp_ms = int(m.group(1))
            utc_dt = datetime.utcfromtimestamp(timestamp_ms / 1000.0)
            start_time = pytz.utc.localize(utc_dt).astimezone(tz)

            print(f"\nStart time from sync_messages.txt: {start_time}")
            return start_time

    # ------------------------------------------------------------
    # 2) Look for metadata files (recursive, unique)
    # ------------------------------------------------------------
    meta_oe = list(session_path.rglob("MetadataOpenephys.txt"))
    meta_legacy = list(session_path.rglob("Metadata.txt"))

    if len(meta_oe) > 1:
        raise RuntimeError(
            f"\nMultiple MetadataOpenephys.txt files found:\n{meta_oe}"
        )

    if len(meta_legacy) > 1:
        raise RuntimeError(
            f"\nMultiple Metadata.txt files found:\n{meta_legacy}"
        )

    metadata_file = None
    source = None

    if meta_oe:
        metadata_file = meta_oe[0]
        source = "MetadataOpenephys.txt"
    elif meta_legacy:
        metadata_file = meta_legacy[0]
        source = "Metadata.txt"

    if metadata_file is not None:
        text = metadata_file.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Software Time.*?:\s*(\d+)", text)

        if not m:
            raise ValueError(
                f"\nNo 'Software Time' timestamp found in {metadata_file}"
            )

        timestamp_ms = int(m.group(1))
        utc_dt = datetime.utcfromtimestamp(timestamp_ms / 1000.0)
        start_time = pytz.utc.localize(utc_dt).astimezone(tz)

        print(f"\nStart time from {source}: {start_time}")
        return start_time

    # ------------------------------------------------------------
    # 3) Fallback: parse date from folder name
    # ------------------------------------------------------------
    print(f"\nNo metadata found. Extracting date from folder name: {session_path.name}")

    try:
        date_str = session_path.name.split("-")[1][:6]
        recording_date = datetime.strptime(date_str, "%y%m%d").date()
        start_time = tz.localize(
            datetime.combine(recording_date, datetime.min.time())
        )
        print(f"\nStart time from folder name (midnight placeholder): {start_time}")
        return start_time
    except Exception as e:
        print(f"\nError extracting date from folder name: {e}")

    # ------------------------------------------------------------
    # 4) Final fallback
    # ------------------------------------------------------------
    print("\nFailed to determine start time. Using placeholder: 2000-01-01 00:00:00")
    return tz.localize(datetime(2000, 1, 1, 0, 0, 0))



def get_matlab_position(pos_file, vbl_name='pos'):
    """
    Loads position from a MATLAB .mat file (either v7.3/HDF5 or older v7),
    ensuring correct format and dimensions, and returns a TsdFrame.

    Parameters
    ----------
    pos_file : str
        Path to the .mat file containing position data.
    vbl_name : str, optional
        Name of the variable in the .mat file that holds the position data.
        Defaults to 'pos'.

    Returns
    -------
    nap.TsdFrame
        A TsdFrame object with columns ['x', 'y'].
    """
    print('Importing position from a .mat file...')

    try:
        # Attempt to open as an HDF5 (v7.3) file
        with h5py.File(pos_file, 'r') as pos_data:
            # Check that vbl_name actually exists in the file
            if vbl_name not in pos_data:
                raise ValueError(f"Variable '{vbl_name}' not found in HDF5 file {pos_file}")

            # Extract position array and time array
            pos = pos_data[vbl_name]['data'][()].T
            pos_index = pos_data[vbl_name]['t'][()].T.squeeze()
            pos_index = np.atleast_1d(pos_index).ravel()

    except OSError:
        # If not an HDF5 file, load using scipy.io (v7 or earlier)
        pos_data = spio.loadmat(pos_file, simplify_cells=True)

        # Make sure the requested variable is present
        if vbl_name not in pos_data:
            raise ValueError(f"Variable '{vbl_name}' not found in MAT-file {pos_file}")

        # Possible older style: a dictionary with 'data' and 't'
        if isinstance(pos_data[vbl_name], dict) and 'data' in pos_data[vbl_name]:
            try:
                pos = np.array(pos_data[vbl_name]['data']).squeeze()
                pos_index = np.atleast_1d(pos_data[vbl_name]['t']).ravel()
            except KeyError as e:
                raise ValueError(
                    f"Expected 'data' and 't' fields in '{vbl_name}' dict, but got {e}"
                )
        else:
            # Otherwise, assume pos_data[vbl_name] is a 2D array with columns = [time, x, y, ...]
            arr = np.array(pos_data[vbl_name])
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(
                    f"'{vbl_name}' must be a 2D array with at least 2 columns ([time, x, y, ...])."
                )
            pos_index = arr[:, 0]
            pos = arr[:, 1:]

    # Finally, build the TsdFrame
    # By default, label columns as ['x', 'y']. If your data has more columns,
    # you can adjust this part accordingly (e.g., ['x', 'y', 'z'] for 3D).
    pos_tsdframe = nap.TsdFrame(t=pos_index, d=pos, columns=['x', 'y'])

    return pos_tsdframe


def get_matlab_hd(hd_file, vbl_name='ang'):
    """
    Loads head-direction data from a MATLAB .mat file,
    ensuring correct format and dimensions.
    """
    print('Importing head-direction from a .mat file...')

    try:
        # Try loading as an HDF5 (v7.3) file
        with h5py.File(hd_file, 'r') as hd_data:
            # Extract and process angle data
            hd = hd_data[vbl_name]['data'][()].T.squeeze()
            hd = np.atleast_1d(hd % (2 * np.pi))  # Ensure 1D
            hd_index = hd_data[vbl_name]['t'][()].T.squeeze()
            hd_index = np.atleast_1d(hd_index).ravel()  # Ensure index is 1D

    except OSError:
        # If not an HDF5 file, load using scipy.io (v7 or earlier)
        hd_data = spio.loadmat(hd_file, simplify_cells=True)
        if 'data' in hd_data[vbl_name]:
            hd = np.array(hd_data[vbl_name]['data']).squeeze()  # Extract inner array
            hd_index = np.atleast_1d(hd_data[vbl_name]['t'].squeeze()).ravel()
        else:
            pos = hd_data[vbl_name][:, 1:]
            pos_index = hd_data[vbl_name][:, 0]

    hd_tsd = nap.Tsd(t=hd_index, d=hd)

    return hd_tsd


def get_matlab_trackdistance(file_name, vbl_name='trackdist'):
    """
    Loads position from a MATLAB .mat file,
    ensuring correct format and dimensions.
    """
    print('Importing track distance from a .mat file...')

    try:
        # Try loading as an HDF5 (v7.3) file
        with h5py.File(file_name, 'r') as file_contents:
            # Extract and process position data
            print('Recent Matlab versions use HDF5 files and need a different loader. Please add code here.')

    except OSError:
        # If not an HDF5 file, load using scipy.io (v7 or earlier)
        file_contents = spio.loadmat(file_name, simplify_cells=True)
        vbl = np.array(file_contents[vbl_name][()]).squeeze()  # Extract inner array and remove unnecessary dimensions
        data = vbl[:, 1]
        index = vbl[:, 0]

    tsd = nap.Tsd(t=index, d=data)

    return tsd


def get_matlab_spikes(path):
    print('Importing spikes and waveforms from .mat files...')

    # file names for spikes, angle, and epoch files
    spike_file = path / 'SpikeData.mat'
    waveform_file = path / 'Waveforms.mat'
    wfeatures_file = path / 'WaveformFeatures.mat'

    # Next lines load the spike data from the .mat file
    spikedata = spio.loadmat(spike_file, simplify_cells=True)
    total_cells = np.arange(0, len(spikedata['S']['C']))
    spikes = {}
    for cell in total_cells:
        timestamps = np.array(spikedata['S']['C'][cell]['tsd']['t'])  # Convert to numpy array
        spikes[cell] = nap.Ts(t=timestamps)  # Create Ts object for each neuron

    # get spike metadata
    waveforms = spio.loadmat(waveform_file, simplify_cells=True)
    waveforms = waveforms['meanWaveforms']
    wfeatures = spio.loadmat(wfeatures_file, simplify_cells=True)
    shank_id = spikedata['shank'] - 1,
    shank_id = shank_id[0]
    maxIx = wfeatures['maxIx']

    spikes = nap.TsGroup(spikes)  # Convert dictionary to TsGroup

    return spikes, waveforms, shank_id


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


def read_nrs(file_path):
    """
    Parses an .nrs (Neuroscope) XML file and extracts relevant metadata.

    Parameters:
    - file_path (str or Path): Path to the .nrs XML file.

    Returns:
    - dict: Parsed metadata from the .nrs file.
    """

    print(f'Importing metadata from {file_path}...')

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

    root = tree.getroot()

    # Initialize data structure
    data = {
        "files": [],
        "displays": [],
        "parameters": {},
        "channels_selected": [],
        "channels_shown": []
    }

    # Extract <files> section
    files_section = root.find("files")
    if files_section is not None:
        for file in files_section.findall("file"):
            data["files"].append({
                "name": file.get("name", "Unknown"),
                "type": file.get("type", "Unknown"),
                "path": file.get("path", "Unknown")
            })

    # Extract <displays> section
    displays_section = root.find("displays")
    if displays_section is not None:
        for display in displays_section.findall("display"):
            data["displays"].append({
                "name": display.get("name", "Unknown"),
                "color": display.get("color", "Unknown"),
                "scale": display.get("scale", "Unknown"),
                "offset": display.get("offset", "Unknown")
            })

    # Extract additional parameters
    for param in root.findall("parameter"):
        key = param.get("name", "unknown_param")
        value = param.text.strip() if param.text else None
        data["parameters"][key] = value

    # Extract channelsSelected (now correctly finding the nested elements)
    channels_selected_section = root.find(".//channelsSelected")  # Use XPath to find it anywhere
    if channels_selected_section is not None:
        data["channels_selected"] = [
            int(ch.text.strip()) for ch in channels_selected_section.findall("channel") if ch.text.strip().isdigit()
        ]

    # Extract channelsShown (now correctly finding the nested elements)
    channels_shown_section = root.find(".//channelsShown")  # Use XPath to find it anywhere
    if channels_shown_section is not None:
        data["channels_shown"] = [
            int(ch.text.strip()) for ch in channels_shown_section.findall("channel") if ch.text.strip().isdigit()
        ]

    return data


def read_metadata_excel(
    file_path,
    file_name,
    print_output=False,
    save_path=None,
    save_name="SessionMetadata.json",
):
    """
    Read an Excel metadata file and structure it into a nested dictionary,
    grouping fields by column-name prefixes, and optionally save it as JSON.

    The function selects exactly one row where the 'file_name' column
    matches `file_name` exactly. Columns of the form 'probe_<n>_<key>'
    are grouped into a list of probe dictionaries under the 'probe' key.
    Other columns are grouped by the prefix before the first underscore.
    """

    file_path = Path(file_path)

    # Load the Excel file
    df = pd.read_excel(file_path)

    # Filter the row where 'file_name' matches exactly
    df = df[df["file_name"] == file_name]

    if df.empty:
        raise ValueError(f"No entry found for file_name: {file_name}")

    if len(df) > 1:
        raise ValueError(
            f"Expected exactly one entry for file_name '{file_name}', "
            f"but found {len(df)} matches:\n"
            f"{df['file_name'].tolist()}"
        )

    df = df.reset_index(drop=True)

    metadata = {}
    probe_data = {}

    probe_pattern = re.compile(r"probe_(\d+)_(\w+)")

    for col in df.columns:
        value = df[col].iloc[0]
        if pd.isna(value):
            value = None
        elif isinstance(value, np.generic):
            value = value.item()

        match = probe_pattern.match(col)
        if match:
            probe_num, probe_key = match.groups()
            probe_num = int(probe_num)
            if probe_num not in probe_data:
                probe_data[probe_num] = {"id": probe_num}
            probe_data[probe_num][probe_key] = value
        else:
            parts = col.split("_", 1)
            if len(parts) == 2:
                group, key = parts
            else:
                group, key = "misc", parts[0]

            if group not in metadata:
                metadata[group] = {}
            metadata[group][key] = value

    # Convert probe_data to list
    metadata["probe"] = list(probe_data.values())

    if print_output:
        pprint.pprint(metadata, width=100)

    # Optional: save metadata as JSON
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if not save_name.lower().endswith(".json"):
            save_name += ".json"

        json_path = save_path / save_name
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata JSON written to: {json_path}")

    return metadata

