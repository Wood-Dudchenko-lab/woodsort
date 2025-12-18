
#from __future__ import annotations
import re
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


def load_neuroscope_channels(recfolder_path):
    """
    Return a DataFrame mapping channel number -> anatomical group index,
    preserving XML order and making row order explicit.

    Columns:
      - channel: int
      - group: int
      - channel_order: int  (row order as encountered in XML)
    """

    recfolder_path = Path(recfolder_path)

    # ------------------------------------------------------------
    # Find XML file
    # ------------------------------------------------------------

    # Select the XML with smallest recording number (robust to paths without "recording###")
    xml_candidates = list(recfolder_path.rglob("continuous.xml"))
    if len(xml_candidates) == 0:
        raise FileNotFoundError("No continuous.xml found under this recording folder.")

    xml_with_index = []
    for p in xml_candidates:
        m = re.search(r"recording(\d+)", str(p))
        rec_idx = int(m.group(1)) if m else float("inf")
        xml_with_index.append((rec_idx, p))

    xml_with_index.sort(key=lambda t: t[0])
    xml_path = xml_with_index[0][1]

    print(f"Using XML file: {xml_path}")

    # ------------------------------------------------------------
    # Parse XML (fallback name swap for hyphen/underscore)
    # ------------------------------------------------------------
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        print(f"Unable to parse XML file: {xml_path}")
        return

    root = tree.getroot()

    rows = []
    group_idx = 0

    for desc in root.findall("anatomicalDescription"):
        for cg in desc.findall("channelGroups"):
            for group in cg.findall("group"):
                for ch in group.findall("channel"):
                    if ch.text is None:
                        continue
                    rows.append(
                        {
                            "channel_0based": int(ch.text),
                            "channel_group": group_idx,
                        }
                    )
                group_idx += 1

    if not rows:
        raise ValueError("No anatomical channel groups found in XML.")

    df = pd.DataFrame(rows)
    df.index.name = 'channel_order'

    return df

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

    # Add annotation about corresponding DAT file acquisition channels
    probe.annotate_contacts(acquisition_channel=xml_channel_indices)

    print("Probe updated with Neuroscope mapping")

    return probe


