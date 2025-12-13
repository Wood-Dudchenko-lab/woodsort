import pynapple as nap
import json
import pandas as pd
from warnings import warn
from spikeinterface.core import SortingAnalyzer, BaseSorting
import numpy as np
from pathlib import Path

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



