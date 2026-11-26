# Version 1.5 #

## Sigima Version 1.5.0 ##

### Requirements ###

* **Third-party**:
  * NiBabel 5.3 or later is an optional dependency, installed with the new `nifti` extra (`pip install sigima[nifti]`): it is required to read NIfTI images.

### ✨ New features since version 1.4.2 ###

* **NIfTI image import**: Single-file NIfTI-1 and NIfTI-2 images (`*.nii` and `*.nii.gz`) can now be read when NiBabel is installed. A 2D image is read directly. For a 3D volume, `NiftiImageImportParam` selects a slice or a range of slices along the third axis of the file, and one image is returned per slice, reading only the selected slices. `get_nifti_image_info` gives the dimensions, data types and spatial unit of a file without reading its voxels. Values are those of the file, scaling factors included, without windowing; pixel sizes and units come from the spatial transform of the file when its slice plane is not sheared, and the import provenance (version, slice index, affine transform, calibration) is stored in the `nifti` metadata of each image. Volumes with four dimensions or more, complex and color images, separate `.hdr`/`.img` files and NIfTI writing are not supported (closes [Issue #152](https://github.com/DataLab-Platform/Sigima/issues/152)).
* **Compound file extensions**: File formats may now declare compound extensions such as `*.nii.gz`. The longest matching extension selects the format, whatever the case of the file name, so that a `*.nii.gz` file is not taken for another format declaring `*.gz`.
