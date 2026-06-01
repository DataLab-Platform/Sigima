# Bug: Detection functions fail with inverse ROI

## Description

When running any detection function (**Blob detection OpenCV/DoG/DoH/LoG**, **Hough circles**, **Peak detection**) on an image with an inverse ROI, the detection either crashes or produces no results.

Two separate problems were identified:

1. **Masked arrays not handled by detection algorithms**: detection functions pass `numpy.ma.MaskedArray` directly to underlying libraries (scikit-image, OpenCV, scipy) that don't support masked arrays.
2. **Wrong bounding box for inverse ROI**: `ImageObj.get_data(roi_index)` crops the data to the bounding box of the ROI **shape** — but for an inverse ROI, the region of interest is the whole image *minus* the shape. The crop returns only the small rectangle around the shape, almost entirely masked, with useful pixels only in the corners between the shape and its bounding rectangle.

## Steps to reproduce

1. Open or create any image with visible features
2. Run **Analysis > Blob detection (OpenCV)** with **"Create ROIs"** enabled
3. ROIs are created around detected blobs
4. Select a ROI and enable **"Inverse ROI logic"**
5. Re-run the detection
6. **Expected**: blobs detected outside the ROI shape
7. **Actual**: no blobs detected, or `RuntimeWarning: invalid value encountered in cast`

## Root cause

### Problem 1: Masked arrays in detection functions

`ImageObj.get_data(roi_index)` returns a `MaskedArray` when a ROI is defined. The detection functions in `sigima/tools/image/detection.py` pass this directly to algorithms that ignore or mishandle the mask:

- **OpenCV** (`find_blobs_opencv`): `rescale_intensity` casts NaN to uint8 → `RuntimeWarning`
- **scikit-image** (`blob_dog/doh/log`): ignores the mask, detects features in masked area
- **scipy** (`get_2d_peaks_coords`, `get_hough_circle_peaks`): same

### Problem 2: Bounding box ignores inverse flag

`get_data(roi_index)` uses `single_roi.get_bounding_box()` to crop the data. This always returns the bounding box of the ROI shape itself. For an inverse ROI (where the *exterior* is the region of interest), this crops to the wrong area — the small rectangle inscribing the shape, where most pixels are masked.

The same issue affects `compute_geometry_from_obj` in `sigima/proc/image/base.py`, which applies a coordinate offset based on `get_bounding_box()` — incorrect for inverse ROIs where the data covers the full image.

## Fix

### 1. `sigima/tools/image/detection.py` — Fill masked values before detection

All affected detection functions now convert masked arrays to regular arrays using `data.filled(np.ma.median(data))`, replacing masked pixels with the median of unmasked data. This provides a neutral background that minimizes detection artifacts at mask boundaries.

Affected functions: `get_2d_peaks_coords`, `get_hough_circle_peaks`, `find_blobs_dog`, `find_blobs_doh`, `find_blobs_log`, `find_blobs_opencv`.

Not modified: `get_contour_shapes` (already handles masked arrays correctly).

### 2. `sigima/objects/image/object.py` — Full-image bounding box for inverse ROI

In `ImageObj.get_data()`, when the ROI has `inverse=True`, the bounding box is now set to the entire image instead of the ROI shape:

```python
if getattr(single_roi, "inverse", False):
    x0, y0, x1, y1 = 0, 0, self.data.shape[1], self.data.shape[0]
else:
    x0, y0, x1, y1 = self.physical_to_indices(single_roi.get_bounding_box(self))
```

Note: `get_bounding_box()` itself was **not** modified because other callers (ROI visualization, coordinate conversion) need the actual shape geometry.

### 3. `sigima/proc/image/base.py` — Post-filter and offset correction

Two changes in `compute_geometry_from_obj`:

- **Post-filter**: after detection returns coordinates (in pixel space), detections whose center falls on a masked pixel are discarded. This catches any remaining false positives from the fill value.
- **Offset skip**: for inverse ROIs, the ROI bounding box offset is skipped since the data already covers the entire image and coordinates are already in the full-image reference frame.

```python
if getattr(single_roi, "inverse", False):
    pass  # No offset needed
else:
    x0, y0, _x1, _y1 = single_roi.get_bounding_box(obj)
    coords[:, colx] += x0 - obj.x0
    coords[:, coly] += y0 - obj.y0
```

## Tests

All 1188 image tests pass.
