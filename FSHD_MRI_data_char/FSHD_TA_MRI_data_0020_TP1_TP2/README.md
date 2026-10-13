# Student MRI sample: tibialis anterior at two visits

This folder contains a small teaching dataset derived from subject UW_0020. It contains eight cross-sectional samples from the **left tibialis anterior** at two visits (TP1 and TP2).

Each sample has three NumPy arrays:

- `TP1_slice_##_water.npy` or `TP2_slice_##_water.npy`: water signal image
- `TP1_slice_##_fat.npy` or `TP2_slice_##_fat.npy`: fat signal image
- `TP1_slice_##_mask.npy` or `TP2_slice_##_mask.npy`: binary tibialis anterior mask

`slice_info.csv` maps each sample to its visit, physical distance, and filenames. Students should calculate fat fraction themselves from the water and fat arrays; a precomputed fat-fraction image is intentionally not included.

The arrays contain the TA-containing half of each in-plane image section. This removes the unrelated side while retaining surrounding anatomy and background. Values retain the original image signal units. The mask contains 1 for TA pixels and 0 elsewhere.

Open `FSHD_TA_MRI_student_notebook.ipynb` from the parent `outputs` folder so the relative data path works as written.
