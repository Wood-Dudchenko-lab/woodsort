import spikeinterface.full as si
import woodsort.neuroscope as neuroscope
import numpy as np
from pprint import pprint
from probeinterface import Probe, ProbeGroup

def process_openephys_with_neuroscope(
    recording,
    probe,
    session_path,
    shank_groups,
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
    probe : probeinterface.Probe or probeinterface.ProbeGroup
        Probe definition to attach to the recording. If a `ProbeGroup` is provided,
        the first probe (`probe.probes[0]`) is used.
    session_path : pathlib.Path or str
        Path to the session directory containing the Neuroscope files used by
        `neuroscope.load_neuroscope_channels(...)`. Also used as the output
        directory when `save_mapping=True`.
    shank_groups : list[int]
        Channel group IDs (as defined by the Neuroscope mapping) to include.
        Only rows whose `channel_group` is in this list are used to build the
        probe-to-recording mapping.
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


    # Validate if probe is a probe not ProbeGroup
    if isinstance(probe, ProbeGroup):
        print('Probe is a ProbeGroup! Using the first probe')
        probe = probe.probes[0]

    # Validate plot_range
    if plot_range is not None:
        if not isinstance(plot_range, (list, tuple)) or len(plot_range) != 2:
            raise ValueError(
                "plot_range must be None or a list/tuple of length 2 "
                "(t_start, t_end)"
            )

    # Load Neuroscope channel mapping
    channel_mapping = neuroscope.load_neuroscope_channels(session_path)

    # Get channel indices corresponding to the probe
    neuroscope_channel_indices = (
        channel_mapping.loc[
            channel_mapping["channel_group"].isin(shank_groups),
            "channel_0based",
        ].tolist()
    )

    # Add neuroscope mapping to probe
    probe = neuroscope.add_neuroscope_mapping(
        probe,
        neuroscope_channel_indices
    )

    #print('Device channel indices and contact ids before adding to the recording')
    #pprint(dict(zip(probe.device_channel_indices, probe.contact_ids)))

    # Attach probe
    recording = recording.set_probe(probe, group_mode="by_shank")

    if plot_probe:
        si.plot_probe_map(recording, with_device_index=True)
        si.plot_probe_map(recording, with_channel_ids=True)

    # check for excluded channels:
    if excluded_channels is not None:

        if not isinstance(excluded_channels, list):
            raise TypeError("excluded_channels must be a list or None")

        excluded_channels_cleaned = []
        for ch in excluded_channels:
            if not isinstance(ch, str):
                raise TypeError(f"Excluded channel {ch} is not a string")

            ch = ch.strip()

            if ch.startswith("CH"):
                # sanity check: CH + digits
                if not ch[2:].isdigit():
                    raise ValueError(f"Invalid channel format: {ch}")
                excluded_channels_cleaned.append(ch)
            else:
                # allow bare numbers like "17"
                if ch.isdigit():
                    excluded_channels_cleaned.append(f"CH{ch}")
                else:
                    raise ValueError(f"Invalid channel format: {ch}")

        recording = recording.remove_channels(excluded_channels_cleaned)
        print(f"Excluded channels: {', '.join(excluded_channels_cleaned)}")

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

    # filter + bad channel removal
    print('Applying high-pass filter (300-6000 Hz)...')
    recording = si.bandpass_filter(recording)

    if remove_bad_channels:
        print('Detecting and removing bad channels...')
        recording = si.detect_and_remove_bad_channels(recording)

        # get IDs of bad channels (0 based)
        bad_ids = []
        for rec_g in recording.values():
            bad = rec_g._kwargs.get("bad_channel_ids")
            if bad is None:
                continue
            for ch in bad:
                # ch is something like 'CH17' or np.str_('CH17')
                bad_ids.append(int(str(ch)[2:]) - 1)

        print(f'Bad channel IDs (0-based): {bad_ids}')

        # mark bad channels in channel mapping
        channel_mapping["is_bad"] = False
        channel_mapping.loc[
            channel_mapping["channel_0based"].isin(bad_ids),
            "is_bad"
        ] = True

    # save channel mapping
    if save_mapping:
        out_path = session_path / "ChannelMapping.csv"
        channel_mapping.to_csv(out_path, index=True)
        print(f'Channel mapping saved to {out_path} \n')

    return recording