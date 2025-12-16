
import numpy as np
import pynapple as nap
import spikeinterface.full as si
from spikeinterface.core import SortingAnalyzer
from neuroconv.tools.spikeinterface import add_sorting_to_nwbfile
import pandas as pd
import json

import matplotlib.pyplot as plt

def get_analyzer_spikes(analyzer: SortingAnalyzer):
    """
    Analyzer-only export.

    Returns
    -------
    spikes : nap.TsGroup
        TsGroup keyed by integer unit ids.
    waveforms : np.ndarray | None
        Templates array (n_units, n_samples, n_channels).
    max_channels : np.ndarray
        0-based channel index of max-amplitude template channel per unit.
    """

    sorting = analyzer.sorting
    unit_ids = list(sorting.unit_ids)

    # Enforce integer TsGroup keys
    try:
        unit_ids_ints = [int(u) for u in unit_ids]
    except Exception:
        unit_ids_ints = list(range(len(unit_ids)))

    # --- spikes ---
    spikes = nap.TsGroup(
        {
            uid_int: nap.Ts(
                t=sorting.get_unit_spike_train(uid, return_times=True)
            )
            for uid_int, uid in zip(unit_ids_ints, unit_ids)
        }
    )

    # --- waveforms (templates) ---
    waveforms = None
    ext_templates = analyzer.get_extension("templates")
    if ext_templates is not None:
        waveforms = ext_templates.get_data()

    # --- max channel per unit (0-based index) ---
    max_channels = si.get_template_extremum_channel(
        analyzer, peak_sign="both", outputs="id"
    )

    max_channels = np.array(
        [int(max_channels[i].replace("CH", "")) for i in range(len(max_channels))],
        dtype=int
    )
    max_channels = max_channels - 1  # 0-based

    return spikes, waveforms, max_channels





