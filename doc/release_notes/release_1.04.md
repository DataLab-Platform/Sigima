# Version 1.4 #

## Sigima Version 1.4.0 ##

### Requirements ###

* Python 3.10 or later is now required. Python 3.9 users must keep an earlier compatible release.

### ✨ New features since version 1.3.0 ###

* **Origin of experimental data**: Objects imported from experimental files or captured from instruments can record where their data came from in the new `experimental_source` metadata entry: source kind, file path or instrument address, scan or dataset, selected part, timestamps, applied conversions and other details. The entry is a versioned dictionary made of JSON types only, so it remains readable without the reader or connector that created it, and it is restored with the same Python types by Sigima's native signal and image files (and by DataLab workspaces): a restored entry can be serialized to JSON or stored again. Values are limited to what HDF5 files restore exactly: `None` and keys that are not plain names are rejected, and NumPy scalars are converted to Python values. It is created and accessed with `make_experimental_source`, `set_experimental_source` and `get_experimental_source` (`sigima.objects`). The existing `source` metadata entry is unchanged.
* **Group UUID from the remote client**: `add_group` of the remote client (`sigima.client`) now returns the UUID of the created group, so that objects can be added to it with `group_id=...` (DataLab ≥ 1.4.0; older versions return `None`).

The [user guide](../user_guide/overview.rst) shows how readers and connectors attach experimental-source metadata, save and restore it, and send an acquisition to a DataLab group. Source recording is opt-in and does not make the experiment reproducible.

### 🛠️ Bug Fixes since version 1.3.0 ###

* **Exact integer import**: Integer data that does not fit any integer type supported by the destination object (for example `uint32` counts above 2 147 483 647 in an image, or large integers in a signal) is now converted to a floating-point type that represents every value exactly. Such data was previously converted to `float32` when the source used 4-byte integers, which rounded counts above 16 777 216. A warning is now emitted when no supported floating-point type is exact (integers beyond 2⁵³).
