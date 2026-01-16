import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pynapple.io.interface_nwb import NWBFile

def test_nwbfile(
    nwbfile,
    t_start: float,
    t_end: float,
    cell_number: int = 1,
    show_tables: bool = False,
):
    """
    Plot anatomically-sorted waveform template and anatomically-sorted stacked LFP
    for a given unit and time window.

    Parameters
    ----------
    rec : object
        Your recording wrapper that contains `rec.nwb` (likely NWBFile loaded via your pipeline).
    nwbfile : pynwb.NWBFile
        The NWB file object. Used for accessing units['waveform_mean'].
    cell_number : int
        Unit index into nwbfile.units['waveform_mean'].
    t_start : float
        Start time in seconds for LFP plot.
    t_end : float
        End time in seconds for LFP plot.
    show_tables : bool
        If True, prints devices and electrodes table.
    verbose : bool
        If True, prints small progress info.

    Returns
    -------
    dict
        Dictionary with useful outputs:
        - electrodes_table_sorted
        - sort_ix
        - waveform (wf)
        - lfp_interval
        - time_axis (t)
    """

    # --- optional: temporarily expand pandas printing ---
    pd_options = {
        "display.max_rows": None,
        "display.max_columns": None,
        "display.width": None,
        "display.max_colwidth": None,
    }
    old_options = {}
    for k, v in pd_options.items():
        old_options[k] = pd.get_option(k)
        pd.set_option(k, v)

    try:

        if type(nwbfile) == NWBFile:
            nwbfile = nwbfile.nwb

        # --- devices ---
        if show_tables:
            print(nwbfile.devices)

        # --- electrode table ---
        electrodes_table = nwbfile.electrodes.to_dataframe()
        if show_tables:
            print(electrodes_table)

        # Anatomical sorting: shank (aggregation_key), then y coordinate top->bottom (rel_y descending)
        required_cols = ["aggregation_key", "rel_y"]
        missing = [c for c in required_cols if c not in electrodes_table.columns]
        if missing:
            raise KeyError(
                f"Electrodes table missing columns {missing}. "
                f"Available columns: {list(electrodes_table.columns)}"
            )

        electrodes_table_sorted = electrodes_table.sort_values(
            by=["aggregation_key", "rel_y"],
            ascending=[True, False],
        )
        sort_ix = electrodes_table_sorted.index.to_numpy().astype(int)

        # --- waveform plot ---
        if "waveform_mean" not in nwbfile.units:
            raise KeyError("nwbfile.units is missing 'waveform_mean' column/dataset.")

        waveforms = nwbfile.units["waveform_mean"][:]  # (units, samples, channels)
        if cell_number < 0 or cell_number >= waveforms.shape[0]:
            raise IndexError(
                f"cell_number={cell_number} out of bounds. "
                f"Valid range: 0..{waveforms.shape[0]-1}"
            )

        wf = waveforms[cell_number, :, sort_ix]

        plt.figure(figsize=(12, 5))
        im = plt.imshow(wf, aspect="auto", origin="lower")
        plt.ylabel("Channel")
        plt.xlabel("Sample")
        plt.title(f"Waveform template for unit {cell_number} (sorted by shank and top to bottom)")
        plt.colorbar(im, label="Amplitude (arb. units)")
        plt.tight_layout()
        plt.show()

        # --- LFP plot ---

        # Path you used:
        # rec.nwb.processing['ecephys']['LFP'].electrical_series['LFP']
        try:
            lfp = nwbfile.processing["ecephys"]["LFP"].electrical_series["LFP"]
        except Exception as e:
            raise KeyError(
                "Could not access LFP at rec.nwb.processing['ecephys']['LFP'].electrical_series['LFP']"
            ) from e

        if not hasattr(lfp, "rate") or lfp.rate is None:
            raise ValueError("LFP series has no sampling rate (lfp.rate is missing/None).")

        if t_end <= t_start:
            raise ValueError(f"t_end must be > t_start (got {t_start=} {t_end=}).")

        start_idx = int(np.floor(t_start * lfp.rate))
        end_idx = int(np.ceil(t_end * lfp.rate))

        if start_idx < 0:
            start_idx = 0

        # Load data slice and reorder channels anatomically
        lfp_interval = lfp.data[start_idx:end_idx, :][:, sort_ix]
        lfp_interval = lfp_interval[:, ::-1]  # flip for plotting orientation

        n_channels = lfp_interval.shape[1]
        t = np.arange(lfp_interval.shape[0]) / lfp.rate + t_start

        plt.figure(figsize=(12, 6))

        offset = 0.0
        for ch in range(n_channels):
            trace = lfp_interval[:, ch]
            plt.plot(t, trace + offset, color="k", linewidth=0.8)
            offset += np.max(np.abs(trace)) * 1.2 if np.any(trace) else 1.0

        plt.xlabel("Time (s)")
        plt.ylabel("Channels (stacked)")
        plt.title(f"LFP (sorted by shank and top to bottom) | {t_start:.3f}–{t_end:.3f}s")
        plt.tight_layout()
        plt.show()

        return

    finally:
        # restore pandas options
        for k, v in old_options.items():
            pd.set_option(k, v)
