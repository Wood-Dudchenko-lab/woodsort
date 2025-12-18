
import json
import pandas as pd
import numpy as np
from pathlib import Path


def get_metadata_openephys(recfolder_path, save_path=None, save_name="MetadataOpenephys.txt"):
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
    recfolder_path : str or Path
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
    recfolder_path = Path(recfolder_path)

    node_path = next(
        (p for p in recfolder_path.rglob("Record Node*") if p.is_dir()),
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
        save_path = recfolder_path

    with open(save_path / save_name, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(f"{msg}\n")

    print(f"Sync metadata written to: {save_path / save_name}")

    return messages

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
    print('\nExtracting recording boundaries...')
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

        print(f"Extracting start and end times: {rec_folder}")

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
    epoch_df = pd.DataFrame(epochs_all, columns=["Start", "End"])

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




