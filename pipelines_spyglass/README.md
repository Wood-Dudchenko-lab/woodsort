# Spyglass spike-sorting & curation pipeline

This directory holds the **Spyglass** spike-sorting pipeline: the sorting runs **inside Spyglass**,
so every stage is stored and tracked as a DataJoint table (full provenance, reproducible
parameters, downstream-ready outputs). Curation is split across several notebooks because the
SpikeInterface GUI cannot run in the Spyglass environment (see
[Why two environments?](#why-two-environments) below).

---

## The Spyglass curation chain

Run the four notebooks in order. Each one's first cells re-derive everything from the same
**"parameters set manually"** block, so keep those parameters identical across all four.

| # | Notebook | Kernel | Produces |
|---|----------|--------|----------|
| 1 | `Pipeline_Spyglass_SpikeSorting.ipynb` | `spyglass` | Sort → `CurationV1` **`curation_id = 0`** (raw), inserted into `SpikeSortingOutput` |
| 2 | `Pipeline_Spyglass_AutomaticCuration.ipynb` | `spyglass` | **Automatic** curation → `curation_id = 1`; **exports** recording + sorting for manual curation |
| 3 | `Pipeline_Spyglass_ManualCuration.ipynb` | `spikeinterface_gui_env` | **Manual** curation in the SpikeInterface GUI → `curation_data.json` |
| 4 | `Pipeline_Spyglass_CompareCurations.ipynb` | `spyglass` | Re-ingests the GUI result → `curation_id = 2`; compares rounds 0/1/2; exposes the chosen one downstream |

```
                          [spyglass env]                                  [spikeinterface_gui_env]
  ┌─────────────────────────────────────────────────────────┐          ┌──────────────────────────────┐
  │ 1. Pipeline_Spyglass_SpikeSorting                        │          │ 3. Pipeline_Spyglass_         │
  │      sort  ──►  CurationV1 curation_id=0 (raw)           │          │    ManualCuration            │
  │                                                          │          │                              │
  │ 2. Pipeline_Spyglass_AutomaticCuration                   │          │   load exported recording +  │
  │      MetricCuration ──► curation_id=1 (automatic)        │          │   sorting ──► SortingAnalyzer │
  │      export recording + sorting  ───────────────────────┼─────────►│   ──► SpikeInterface GUI      │
  │                                                          │  export  │   ──► curation_data.json      │
  │ 4. Pipeline_Spyglass_CompareCurations                    │◄─────────┼───────────────────────────────┘
  │      ingest curation_data.json ──► curation_id=2 (manual)│  json
  │      compare rounds 0 / 1 / 2                            │
  │      expose chosen curation ──► SpikeSortingOutput       │
  └─────────────────────────────────────────────────────────┘

  Handoff folder (shared by steps 2 → 3 → 4):
      manual_curation_export/<sorting_id>/
          recording/                 (binary, written by step 2)
          sorting/                   (npz, written by step 2)
          sorting_analyzer/          (built by step 3)
              spikeinterface_gui/curation_data.json   (written by the GUI in step 3)
```

If you only care about **automatic** curation, you can stop after step 2 and choose `curation_id =
1` as your downstream result. Steps 3 and 4 are only needed for **manual** curation.

## Why two environments?

The SpikeInterface GUI needs a `SortingAnalyzer`, which only exists in SpikeInterface 0.101+.

- The **`spyglass`** environment ships **SpikeInterface 0.99** (older `WaveformExtractor` API) and
  does **not** have `spikeinterface-gui` installed. It runs notebooks 1, 2, and 4.
- The **`spikeinterface_gui_env`** environment has **SpikeInterface 0.104** + `spikeinterface-gui`.
  It runs only notebook 3.

Because the two cannot share a process, step 2 **exports** the recording and sorting to portable
on-disk folders (binary recording + npz sorting), step 3 loads them in the GUI environment and
writes `curation_data.json`, and step 4 reads that file back into Spyglass. The shared handoff
location is `manual_curation_export/<sorting_id>/` (see the diagram).

> **Fallback if the exported folders fail to load in step 3** (a SpikeInterface 0.99 → 0.104
> version mismatch): read the Spyglass recording/sorting directly from their NWB files instead,
> using `spikeinterface.extractors.NwbRecordingExtractor` / `NwbSortingExtractor`, then build the
> analyzer from those.
