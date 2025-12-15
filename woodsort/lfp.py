import json
import numpy as np
from scipy.signal import firwin, fftconvolve
from tqdm import tqdm
import pynapple as nap
from pathlib import Path
import platform
import subprocess
import re
import shutil


def make_lfp_from_dat(
    dat_file,
    output_folder,
    lfp_name,
    input_sample_rate,
    num_channels,
    output_sample_rate=1250,
    lopass=450,
    chunk_size=100_000,
    max_filter_taps=2000
):
    dat_file = Path(dat_file)
    if not dat_file.exists():
        raise FileNotFoundError(f"DAT file not found: {dat_file}")
    if dat_file.name != "continuous.dat":
        raise ValueError(f"Expected a file named 'continuous.dat', got: {dat_file.name}")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    lfp_file = output_folder / f"{lfp_name}.lfp"

    downsample_ratio = int(input_sample_rate / output_sample_rate)

    signal = nap.load_eeg(
        str(dat_file),
        n_channels=num_channels,
        frequency=input_sample_rate,
        channel=None
    )
    total_samples = signal.shape[0]

    # Filter setup (centred FIR)
    ntbuff = int(np.round((4 * input_sample_rate) / lopass))
    ntbuff += (downsample_ratio - ntbuff % downsample_ratio) % downsample_ratio
    ntbuff = min(ntbuff, max_filter_taps)
    fir_coeff = firwin(ntbuff, lopass / (input_sample_rate / 2))

    # Align chunk size to downsample grid
    chunk_size += (downsample_ratio - chunk_size % downsample_ratio) % downsample_ratio
    nb_full_chunks = total_samples // chunk_size
    remainder = total_samples % chunk_size

    with lfp_file.open("wb") as flfp:

        # -------- FULL CHUNKS --------
        for ibatch in tqdm(range(nb_full_chunks), desc="Extracting LFP"):
            chunk_raw_start = ibatch * chunk_size
            chunk_raw_end   = (ibatch + 1) * chunk_size

            window_start = max(0, chunk_raw_start - ntbuff)
            window_end   = min(total_samples, chunk_raw_end + ntbuff)

            dat = signal[window_start:window_end, :].astype(np.float64)

            ds_chunk_size = chunk_size // downsample_ratio
            # Global raw indices we want in the output for this chunk
            raw_idx = chunk_raw_start + np.arange(ds_chunk_size) * downsample_ratio
            local_idx = raw_idx - window_start  # indices into `dat` / `filtered`

            lfp_data = np.zeros((num_channels, ds_chunk_size), dtype=np.int16)

            for ch in range(num_channels):
                filtered = fftconvolve(dat[:, ch], fir_coeff, mode="same")

                # Safety: ensure indices are in-bounds
                if local_idx[-1] >= filtered.shape[0] or local_idx[0] < 0:
                    raise RuntimeError(
                        f"Indexing out of bounds in chunk {ibatch}: "
                        f"local_idx range {local_idx[0]}..{local_idx[-1]} "
                        f"but filtered len {filtered.shape[0]}"
                    )

                lfp_data[ch, :] = filtered[local_idx].astype(np.int16)

            flfp.write(lfp_data.T.tobytes())

        # -------- REMAINDER --------
        if remainder > 0:
            rem_start = nb_full_chunks * chunk_size
            true_ds = remainder // downsample_ratio

            # Window: include left buffer so convolution is valid at rem_start
            window_start = max(0, rem_start - ntbuff)
            window_end   = total_samples

            dat = signal[window_start:window_end, :].astype(np.float64)

            # Global raw indices we want in the remainder output
            raw_idx = rem_start + np.arange(true_ds) * downsample_ratio
            local_idx = raw_idx - window_start

            lfp_data = np.zeros((num_channels, true_ds), dtype=np.int16)

            for ch in range(num_channels):
                filtered = fftconvolve(dat[:, ch], fir_coeff, mode="same")

                if true_ds > 0:
                    if local_idx[-1] >= filtered.shape[0] or local_idx[0] < 0:
                        raise RuntimeError(
                            "Indexing out of bounds in remainder: "
                            f"local_idx range {local_idx[0]}..{local_idx[-1]} "
                            f"but filtered len {filtered.shape[0]}"
                        )

                    lfp_data[ch, :] = filtered[local_idx].astype(np.int16)

            flfp.write(lfp_data.T.tobytes())


def merge_lfp_files(session_folder, output_name="continuous.lfp"):
    session_folder = Path(session_folder)
    out_path = session_folder / output_name

    # Collect component LFP files ONLY
    files = list(session_folder.glob("continuous_*.lfp"))
    files = [f for f in files if f.resolve() != out_path.resolve()]
    if not files:
        raise FileNotFoundError(f"No continuous_*.lfp in {session_folder}")

    # Numeric sort by trailing index X
    def idx(p: Path):
        m = re.search(r"continuous_(\d+)\.lfp$", p.name)
        return int(m.group(1)) if m else 10**12

    files = sorted(files, key=idx)

    system = platform.system().lower()

    # ---------- MERGE ----------
    if system.startswith("windows"):
        expr = " + ".join([f'"{str(f)}"' for f in files])
        cmd = f'copy /B {expr} "{str(out_path)}"'
        subprocess.run(["cmd", "/c", cmd], check=True)
    else:
        quoted_in = " ".join([subprocess.list2cmdline([str(f)]) for f in files])
        quoted_out = subprocess.list2cmdline([str(out_path)])
        cmd = f"cat {quoted_in} > {quoted_out}"
        subprocess.run(["/bin/sh", "-c", cmd], check=True)

    print(f"Merged {len(files)} LFP files into {out_path}.")
    for f in files:
        f.unlink()
    print("Component LFP files removed.")

    return

def copy_xml_to_session(recfolder_path):
    """
    Find a continuous.xml file somewhere under recfolder_path and copy it
    to the top-level session folder.

    The XML chosen is the one with the smallest recording number.
    """

    recfolder_path = Path(recfolder_path)

    # ------------------------------------------------------------
    # Find XML files
    # ------------------------------------------------------------
    xml_candidates = list(recfolder_path.rglob("continuous.xml"))
    if not xml_candidates:
        print(f"No XML file found in {recfolder_path}.")
        return

    # ------------------------------------------------------------
    # Extract recording indices
    # ------------------------------------------------------------
    recording_numbers = []
    for path in xml_candidates:
        match = re.search(r"recording(\d+)", str(path))
        if match:
            recording_numbers.append(int(match.group(1)))
        else:
            recording_numbers.append(float("inf"))

    # ------------------------------------------------------------
    # Select XML with smallest recording index
    # ------------------------------------------------------------
    xml_path = xml_candidates[
        recording_numbers.index(min(recording_numbers))
    ]

    dest_path = recfolder_path / "continuous.xml"

    # ------------------------------------------------------------
    # Copy (overwrite if exists)
    # ------------------------------------------------------------
    shutil.copy2(xml_path, dest_path)
    print(f"Copied XML file.")

    return

def extract_lfp_openephys(
    session_folder,
    output_sample_rate=1250,
    lopass=450,
    chunk_size=100_000,
):
    """
    Find all recording* folders under session_folder. For each recording folder:
      - read structure.oebin in that folder
      - find continuous.dat somewhere underneath
      - process it into session_folder/local_field_potential_X.lfp
    """

    session_folder = Path(session_folder)
    session_folder.mkdir(parents=True, exist_ok=True)

    # Find recording* directories anywhere under the session folder
    recording_folders = sorted(
        [p for p in session_folder.rglob("recording*") if p.is_dir()],
        key=lambda p: (len(p.parts), p.name),
    )

    if not recording_folders:
        print(f"No recording* folders found under {session_folder}")
        return

    idx = 0
    for rec in recording_folders:
        oebin_path = rec / "structure.oebin"
        if not oebin_path.exists():
            # not a real Open Ephys recording folder (or incomplete)
            continue

        dat_files = sorted(rec.rglob("continuous.dat"))
        if not dat_files:
            print(f"\nSkipping {rec}: no continuous.dat found under it.")
            continue

        if len(dat_files) > 1:
            # pick the one closest to the recording folder (least nested)
            dat_files = sorted(dat_files, key=lambda p: len(p.relative_to(rec).parts))

        dat_file = dat_files[0]

        with oebin_path.open("r") as f:
            oebin = json.load(f)

        try:
            cont0 = oebin["continuous"][0]
            input_sample_rate = cont0["sample_rate"]
            num_channels = cont0["num_channels"]
        except Exception as e:
            raise RuntimeError(f"Could not parse metadata from {oebin_path}: {e}")

        idx += 1
        lfp_name = f"continuous_{idx:02d}"
        out_lfp = session_folder / f"{lfp_name}.lfp"

        print(f"\n[{idx}] recording folder: {rec}")

        make_lfp_from_dat(
            dat_file=dat_file,
            output_folder=session_folder,
            lfp_name=lfp_name,
            input_sample_rate=input_sample_rate,
            num_channels=num_channels,
            output_sample_rate=output_sample_rate,
            lopass=lopass,
            chunk_size=chunk_size,
        )
    print(f"\nDone. Processed {idx} recording folder(s).")

    # merge and delete LFP files
    merge_lfp_files(session_folder)
    # copy XML file if present
    copy_xml_to_session(session_folder)

    print(f"LFP extraction done.\n")


