import re
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

def add_neuroscope_mapping(probe, xml_channel_indices):
    # Sanity check for number of channels
    if probe.get_contact_count() != len(xml_channel_indices):
        print("Number of Neuroscope channels doesn't match the SpikeInterface probe!")  # Sanity check for xml channel indices
        print("Check Neuroscope mapping")  # Sanity check for xml channel indices

    # Sort contact positions based on shank IDs and y-coordinates
    shank_ids = probe.shank_ids
    contact_positions = probe.contact_positions
    unique_shank_ids = np.unique(shank_ids)

    # sort channels based on their coordinates, top to bottom and shank-wise
    sorted_coordinates_by_shank = []
    for unique_id in unique_shank_ids:
        id_indices = np.where(shank_ids == unique_id)[0]
        coors = contact_positions[id_indices]
        coors = coors[coors[:, 1].argsort()[::-1]]
        sorted_coordinates_by_shank.append(coors)
    final_sorted_coordinates = np.vstack(sorted_coordinates_by_shank)

    # Update probe with sorted coordinates
    probe.set_contacts(
        final_sorted_coordinates,
        shapes=probe.contact_shapes,
        shape_params=probe.contact_shape_params,
        plane_axes=probe.contact_plane_axes,
        contact_ids=np.arange(len(xml_channel_indices)),
        shank_ids=np.sort(probe.shank_ids),  # Sorted shank_ids
    )

    # Set device channel indices
    probe.set_device_channel_indices(xml_channel_indices)

    print("Probe updated with Neuroscope mapping")

    return probe


def load_neuroscope_channels(recfolder_path, shank_groups=None):
    """
    Locate the session's continuous.xml file, extract anatomical channel groups,
    and return a 1D array of channel indices for the specified shank groups.

    Parameters
    ----------
    recfolder_path : str or Path
        Path to the session folder containing recording* subfolders.
    shank_groups : list, array, or None
        Indices of anatomical groups to extract.
        Must be specified; if None, an error is raised.

    Returns
    -------
    xml_channel_indices : np.ndarray
        Flattened array of channel indices belonging to the selected shanks.
    """

    if shank_groups is None:
        raise ValueError(
            "shank_groups must be specified (e.g. [0], [0,1], etc.) "
            "because this function only returns channel indices."
        )

    recfolder_path = Path(recfolder_path)

    # ------------------------------------------------------------
    # Find XML file
    # ------------------------------------------------------------
    xml_candidates = list(recfolder_path.rglob("continuous.xml"))
    if len(xml_candidates) == 0:
        raise FileNotFoundError("No continuous.xml found under this recording folder.")

    # Select the XML with smallest recording number
    xml_path = sorted(
        xml_candidates,
        key=lambda x: int(re.search(r"recording(\\d+)", str(x)).group(1))
    )[0]

    print(f"Using XML file: {xml_path}")

    # ------------------------------------------------------------
    # Parse XML (fallback name swap for hyphen/underscore)
    # ------------------------------------------------------------
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        alt_name = (xml_path.name.replace("-", "_")
                    if "-" in xml_path.name
                    else xml_path.name.replace("_", "-"))
        alt_path = xml_path.parent / alt_name
        print(f"Trying alternative XML filename: {alt_path}")
        tree = ET.parse(alt_path)

    root = tree.getroot()

    # ------------------------------------------------------------
    # Extract anatomical channel groups ONLY
    # ------------------------------------------------------------
    anatomical_groups = []

    for desc in root.findall("anatomicalDescription"):
        for cg in desc.findall("channelGroups"):
            for group in cg.findall("group"):
                channels = [int(ch.text) for ch in group.findall("channel")]
                if channels:
                    anatomical_groups.append(np.array(channels))

    if len(anatomical_groups) == 0:
        raise ValueError("No anatomical channel groups found in XML.")

    anatomical_groups = np.array(anatomical_groups, dtype=object)

    # ------------------------------------------------------------
    # Select shanks
    # ------------------------------------------------------------
    try:
        xml_channel_indices = np.concatenate(anatomical_groups[shank_groups])
    except IndexError:
        raise IndexError(
            f"shank_groups {shank_groups} exceed available groups "
            f"(found {len(anatomical_groups)})"
        )

    return xml_channel_indices