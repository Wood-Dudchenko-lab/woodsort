import h5py
import os
from pynwb import NWBHDF5IO

def change_genotype(nwb_file_path, new_info):
    """Modify the genotype field in an NWB file safely."""

    # Check if file exists
    if not os.path.exists(nwb_file_path):
        print(f"Error: File '{nwb_file_path}' not found.")
        return

    # Check if new_info is a string
    if not isinstance(new_info, str):
        print("Error: The new genotype must be a string.")
        return

    try:
        with h5py.File(nwb_file_path, "r+") as f:
            # Ensure subject information exists
            if "/general/subject" not in f:
                print("Error: '/general/subject' group not found in NWB file.")
                return

            subject_group = f["/general/subject"]

            # Modify the genotype field
            if "genotype" in subject_group:
                del subject_group["genotype"]  # Remove existing genotype
            subject_group.create_dataset("genotype", data=new_info)  # Add new genotype
            print("Updated! Verifying...")

    except Exception as e:
        print(f"Error while modifying NWB file: {e}")
        return

    # Open NWB file to verify changes
    try:
        with NWBHDF5IO(nwb_file_path, "r") as io:
            nwbfile = io.read()
            if nwbfile.subject and nwbfile.subject.genotype == new_info:
                print("Verification successful. New genotype:", nwbfile.subject.genotype)
            else:
                print("Warning: Genotype modification may not have been applied correctly.")
    except Exception as e:
        print(f"Error while verifying NWB file: {e}")



