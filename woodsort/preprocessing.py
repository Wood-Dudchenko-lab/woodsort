import spikeinterface.full as si
import woodsort.neuroscope as neuroscope
import numpy as np
from pprint import pprint

def process_openephys_with_neuroscope(
    recording,
    probe,
    session_path,
    shank_groups,
    plot_probe=False,
    plot_range=None,
    save_mapping=True
):
    """
    Load Neuroscope XML channel mapping, attach probe to recording,
    group by shank, optionally plot, concatenate, split, and filter.
    """
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

    print('Device channel indices and contact ids before adding to the recording')
    #print(probe.device_channel_indices)
    #print('Channel ids before adding to the recording')
    #print(probe.contact_ids)
    pprint(dict(zip(probe.device_channel_indices, probe.contact_ids)))

    # Sort channel IDs so they are labelled in descending fashion (RESETS AFTER PROBE ADDED TO REC):
    #probe.set_contact_ids(np.sort(probe.contact_ids.astype(int)).astype(str))
    #print(probe.contact_ids)

    # Attach probe
    recording = recording.set_probe(probe, group_mode="by_shank")
    p = recording.get_probe()
    print('Device channel indices and contact ids after adding to the recording')
    #print(p.device_channel_indices)
    #print('Channel ids after adding to the recording')
    #print(p.contact_ids)
    pprint(dict(zip(probe.device_channel_indices, probe.contact_ids)))

    #print(p.contact_ids)

    if plot_probe:
        si.plot_probe_map(recording, with_device_index=True)
        si.plot_probe_map(recording, with_channel_ids=True)

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
    recording = si.bandpass_filter(recording)
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