
import re
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from probeinterface import ProbeGroup

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
        raise ValueError(
            "Number of Neuroscope channels doesn't match the SpikeInterface "
            f"probe: probe has {probe.get_contact_count()} contacts but "
            f"Neuroscope mapping has {len(xml_channel_indices)} channels. "
            "Check shank_groups and the probe layout."
        )

    #TODO: Add handling of probegroup (multiple probes)
    if isinstance(probe, ProbeGroup):
        probe = probe.probes[0]

    # Sort contact positions based on shank IDs and y-coordinates
    contact_count = probe.get_contact_count()
    shank_ids = probe.shank_ids
    if shank_ids is None:
        shank_ids = np.zeros(contact_count, dtype=int)
    else:
        shank_ids = np.asarray(shank_ids)
        if shank_ids.ndim == 0 or shank_ids.size == 1:
            shank_ids = np.repeat(shank_ids.item(), contact_count)
        elif shank_ids.size != contact_count:
            raise ValueError(
                "Probe shank_ids must contain one value per contact: "
                f"got {shank_ids.size} shank IDs for {contact_count} contacts."
            )

    contact_positions = np.asarray(probe.contact_positions)
    unique_shank_ids = np.unique(shank_ids)

    # sort channels based on their coordinates, top to bottom and shank-wise
    sorted_indices_by_shank = []
    for unique_id in unique_shank_ids:
        id_indices = np.where(shank_ids == unique_id)[0]
        coors = contact_positions[id_indices]
        sorted_indices_by_shank.append(id_indices[coors[:, 1].argsort()[::-1]])
    sorted_indices = np.concatenate(sorted_indices_by_shank)
    final_sorted_coordinates = contact_positions[sorted_indices]

    def _reorder_contact_property(values):
        if values is None:
            return None
        values = np.asarray(values, dtype=object)
        if values.ndim == 0:
            return np.repeat(values.item(), contact_count)
        if len(values) != contact_count:
            return values
        return values[sorted_indices]

    # Update probe with sorted coordinates
    probe.set_contacts(
        final_sorted_coordinates,
        shapes=_reorder_contact_property(probe.contact_shapes),
        shape_params=_reorder_contact_property(probe.contact_shape_params),
        plane_axes=_reorder_contact_property(probe.contact_plane_axes),
        contact_ids=np.arange(len(xml_channel_indices)),
        shank_ids=shank_ids[sorted_indices],
    )

    # Set device channel indices
    probe.set_device_channel_indices(xml_channel_indices)

    print("Probe updated with Neuroscope mapping")

    return probe

