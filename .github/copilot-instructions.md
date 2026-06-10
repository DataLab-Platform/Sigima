# Sigima AI Coding Agent Instructions

This document provides essential guidance for AI coding agents working on the Sigima codebase—the computation engine powering DataLab's signal and image processing.

## Project Overview

**Sigima** is a **headless Python library** providing scientific computation functions for 1D signals and 2D images. It is **GUI-independent** (no Qt/PlotPyStack) and designed for testability, modularity, and remote execution.

### Core Architecture: Three-Layer Model

Sigima separates concerns into three distinct layers:

1. **`sigima.objects`**: Data model (`SignalObj`, `ImageObj` wrapping NumPy arrays)
2. **`sigima.proc`**: High-level computation functions operating on objects
3. **`sigima.tools`**: Low-level NumPy algorithms (used by `proc` and external projects)

```python
# Example: Processing a signal object (high-level)
from sigima import SignalObj
import sigima.proc.signal as sips

obj = SignalObj.create(x, y)
result = sips.normalize(obj, sigima.params.NormalizeParam.create(method="minmax"))

# Example: Using low-level tools directly (NumPy arrays)
from sigima.tools.signal import filtering
filtered_y = filtering.apply_moving_average(y, n=5)
```

**Key Design Principle**: `sigima.tools` fills gaps in NumPy/SciPy/scikit-image, not a general-purpose replacement. DataLab uses many `tools` functions independently of the object model.

### Technology Stack

- **Python**: 3.9+ (`from __future__ import annotations`)
- **Core**: NumPy (≥1.22), SciPy (≥1.10.1), scikit-image (≥0.19.2), pandas (≥1.4)
- **GUI Parameters**: guidata (≥3.13) for `DataSet` parameter classes
- **Optional**: opencv-python-headless (≥4.8.1.78)
- **Testing**: pytest with `--gui` flag for visual validation
- **Linting**: Ruff (preferred), Pylint
- **Docs**: Sphinx with French translations (sphinx-intl)

### Workspace Structure

```
Sigima/
├── sigima/
│   ├── objects/          # Data model (SignalObj, ImageObj, ROI)
│   │   ├── signal/       # SignalObj implementation
│   │   ├── image/        # ImageObj implementation
│   │   └── scalar/       # GeometryResult, TableResult
│   ├── proc/             # High-level computation functions
│   │   ├── base.py       # Common processing (ROI, arithmetic)
│   │   ├── signal/       # Signal processing (filtering, FFT, fitting, etc.)
│   │   ├── image/        # Image processing (edges, morphology, detection)
│   │   └── decorator.py  # @computation_function decorator
│   ├── tools/            # Low-level NumPy algorithms
│   │   ├── signal/       # Signal algorithms (peak detection, stability, etc.)
│   │   └── image/        # Image algorithms (detection, geometry, etc.)
│   ├── io/               # I/O for signals/images (CSV, formats)
│   ├── viz/              # Visualization utilities (PlotPy and Matplotlib backends)
│   ├── data/             # Bundled test/example data files
│   ├── params.py         # Centralized parameter import (re-exports from proc/)
│   ├── client/           # SimpleRemoteProxy for DataLab control
│   └── tests/            # pytest test suite
├── scripts/
│   └── run_with_env.py   # Environment loader (.env support)
├── .env                  # PYTHONPATH=.;../guidata;../plotpy
└── pyproject.toml
```

**Related Projects** (sibling directories):
- `../DataLab/` - GUI application using Sigima
- `../PlotPy/` - Plotting library (used in tests with `--gui`)
- `../guidata/` - Parameter/configuration framework

## Development Workflows

### Running Commands

**ALWAYS use `scripts/run_with_env.py`** to load `.env` before running Python commands:

```powershell
# ✅ CORRECT
python scripts/run_with_env.py python -m pytest

# ❌ WRONG - Misses local guidata/plotpy
python -m pytest
```

### Testing

```powershell
# Run all tests (fast, no GUI)
python scripts/run_with_env.py python -m pytest --ff

# Run GUI-assisted validation tests (visual checks)
python scripts/run_with_env.py python -m pytest --gui

# Run specific test module
python scripts/run_with_env.py python -m pytest sigima/tests/signal/processing_unit_test.py

# Coverage
python scripts/run_with_env.py python -m coverage run -m pytest sigima
python -m coverage html
```

**Test Organization**:
- `tests/common/`: ROI, validation, worker, title formatting
- `tests/signal/`: Signal processing tests
- `tests/image/`: Image processing tests
- `tests/io/`: I/O format tests

**Pytest Configuration** (`conftest.py`):
- `env.execenv.unattended = True` (no GUI by default)
- `set_validation_mode(ValidationMode.STRICT)` for tests
- Custom flag: `--gui` enables visual validation

### Linting and Formatting

```powershell
# Ruff (preferred)
python scripts/run_with_env.py python -m ruff format
python scripts/run_with_env.py python -m ruff check --fix

# Pylint
python scripts/run_with_env.py python -m pylint sigima \
    --disable=duplicate-code,fixme,too-many-arguments, \
    too-many-branches,too-many-instance-attributes
```

### Translations

```powershell
# Scan and update .po files
python scripts/run_with_env.py python -m guidata.utils.translations scan \
    --name sigima --directory . --copyright-holder "DataLab Platform Developers" \
    --languages fr

# Compile .mo files
python scripts/run_with_env.py python -m guidata.utils.translations compile \
    --name sigima --directory .
```

## Core Patterns

### 1. Computation Functions with `@computation_function` Decorator

**All `sigima.proc` functions** use this decorator to enable dual calling conventions:

```python
from sigima.proc.decorator import computation_function
import sigima.params

@computation_function()
def my_processing(src: SignalObj, p: MyParam) -> SignalObj:
    """Process signal with my algorithm.

    Args:
        src: Input signal
        p: Processing parameters

    Returns:
        Processed signal
    """
    dst = src.copy()
    # ... processing logic using p.param1, p.param2 ...
    return dst
```

**Dual calling style enabled by decorator**:

```python
# Style 1: DataSet parameter object (DataLab GUI style)
param = sigima.params.MyParam.create(param1=10, param2="value")
result = my_processing(src, param)

# Style 2: Expanded keyword arguments (script-friendly)
result = my_processing(src, param1=10, param2="value")
```

**Key Rules**:
- Parameter class MUST be a `guidata.dataset.DataSet` subclass
- Always re-export parameter classes in `sigima.params` module
- Export computation functions in `__all__` of their module AND in `sigima/proc/{signal|image}/__init__.py`

### 2. Object Model: `SignalObj` and `ImageObj`

**Core attributes**:

```python
# SignalObj
signal.x           # X coordinates (1D NumPy array, float64)
signal.y           # Y data (1D NumPy array, float64)
signal.dx, signal.dy  # Optional uncertainties
signal.xydata      # Property returning (x, y) tuple
signal.set_xydata(x, y, dx=None, dy=None)

# ImageObj
image.data         # 2D NumPy array (various dtypes)
image.x0, image.y0, image.dx, image.dy  # Pixel coordinates
image.metadata     # Dict for labels, units, etc.

# Common to both
obj.roi            # List of ROI objects (SegmentROI, RectangularROI, etc.)
obj.get_data(roi_index=None)  # Extract data with optional ROI mask
obj.copy()         # Deep copy with metadata
```

**Data type enforcement**:
- `SignalObj`: Automatically converts integer X/Y arrays to `float64` for computational precision
- `ImageObj`: Preserves original dtype (uint8, uint16, float32, etc.) for image operations

### 3. Parameter Classes: `guidata.dataset.DataSet`

**All computation parameters** inherit from `guidata.dataset.DataSet`:

```python
import guidata.dataset as gds

class MyParam(gds.DataSet):
    """My processing parameters."""

    param1 = gds.IntItem("Parameter 1", default=10, min=1, max=100)
    param2 = gds.ChoiceItem("Method", ["method1", "method2"], default="method1")

    @staticmethod
    def create(param1: int = 10, param2: str = "method1") -> MyParam:
        """Factory method for easy instantiation."""
        return MyParam(param1=param1, param2=param2)
```

**Conventions**:
- Always provide `create()` static method for script-friendly instantiation
- Export in `sigima.params` for centralized import
- Use descriptive docstrings (shown in DataLab GUI)

### 4. Title Formatting System

**Computation results need titles**. Sigima provides a configurable system:

```python
from sigima.proc.title_formatting import TitleFormatter, FormatResultTitle

class MyParam(gds.DataSet):
    # ... parameter definitions ...

    def generate_title(self) -> str:
        """Generate human-readable title for this computation."""
        return f"my_processing(p1={self.param1}, p2={self.param2})"

# In computation function
@computation_function()
def my_processing(src: SignalObj, p: MyParam) -> SignalObj:
    dst = src.copy()
    # ... processing ...
    FormatResultTitle.apply(dst, src, p)  # Automatically formats title
    return dst
```

**Title formatting modes**:
- **Parameter mode**: Default, uses `param.generate_title()` → `"normalize[minmax]"`
- **Function mode**: Used by DataLab, shows function name → `"Normalize"`

### 5. ROI (Region of Interest) System

**Types of ROI**:
- **Signal**: `SegmentROI` (X interval)
- **Image**: `RectangularROI`, `CircularROI`, `PolygonalROI`

**Using ROIs in processing**:

```python
# Get data masked by ROI
data = obj.get_data(roi_index=0)  # First ROI
data = obj.get_data()  # All data (no ROI)

# ROI iteration
for roi_index, roi in enumerate(obj.roi):
    masked_data = obj.get_data(roi_index)
    # ... process masked_data ...
```

**ROI creation in detection functions**:

```python
from sigima.objects import create_image_roi_around_points

# Automatically create ROIs around detected features
coords = detect_peaks(image.data)  # Returns N×2 array
rois = create_image_roi_around_points(coords, image,
                                       relative_size=1.5)
result.roi = rois  # Attach to result
```

## Common Tasks

### Adding a New Signal Processing Function

**Complete workflow**:

1. **Implement in `sigima/proc/signal/processing.py` (or appropriate module)**:

```python
from sigima.proc.decorator import computation_function
import sigima.params

@computation_function()
def my_feature(src: SignalObj, p: MyFeatureParam) -> SignalObj:
    """Apply my feature to signal.

    Args:
        src: Input signal
        p: Feature parameters

    Returns:
        Processed signal
    """
    dst = src.copy()
    # Processing logic using src.x, src.y
    dst.y = apply_my_algorithm(src.y, p.threshold)
    FormatResultTitle.apply(dst, src, p)
    return dst
```

2. **Define parameter class in same file**:

```python
class MyFeatureParam(gds.DataSet):
    """Parameters for my feature."""
    threshold = gds.FloatItem("Threshold", default=0.5, min=0, max=1)

    @staticmethod
    def create(threshold: float = 0.5) -> MyFeatureParam:
        return MyFeatureParam(threshold=threshold)

    def generate_title(self) -> str:
        return f"my_feature(thresh={self.threshold})"
```

3. **Export in `sigima/proc/signal/__init__.py`**:

```python
from sigima.proc.signal.processing import my_feature, MyFeatureParam

__all__ = [
    # ... existing exports ...
    "my_feature",
    "MyFeatureParam",
]
```

4. **Re-export parameter in `sigima/params.py`**:

```python
from sigima.proc.signal import MyFeatureParam

__all__ = [
    # ... existing params ...
    "MyFeatureParam",
]
```

5. **Add tests in `sigima/tests/signal/`**:

```python
import sigima.proc.signal as sips
import sigima.params
from sigima.tests.data import get_test_signal

@pytest.mark.validation
def test_my_feature():
    """Test my_feature processing."""
    src = get_test_signal("paracetamol.txt")
    p = sigima.params.MyFeatureParam.create(threshold=0.5)
    result = sips.my_feature(src, p)

    assert result is not None
    assert len(result.y) == len(src.y)
    # Add assertions checking result correctness
```

6. **Document in Sphinx** (if public API):

```python
# Docstring already makes it appear in API docs
# Add usage example in doc/examples/ if complex
```

### Adding Low-Level NumPy Functions to `tools`

**When to use `tools` vs `proc`**:
- Use `tools` for **pure NumPy/SciPy algorithms** that don't need object context
- Use `proc` for functions that need **metadata, ROI, or object operations**

**Example**:

```python
# sigima/tools/signal/myalgorithm.py
import numpy as np
from sigima.tools.checks import check_1d_array

def my_numpy_function(y: np.ndarray, threshold: float) -> np.ndarray:
    """Low-level algorithm operating on NumPy arrays.

    Args:
        y: 1D NumPy array
        threshold: Processing threshold

    Returns:
        Processed array
    """
    check_1d_array(y)  # Input validation
    # ... pure NumPy processing ...
    return result
```

**Export in `sigima/tools/signal/__init__.py`** and document intended usage.

### Handling Integer Signal Data

**Issue**: Integer arrays cause precision loss in computations.

**Solution**: Sigima automatically converts to `float64`:

```python
# This is handled automatically now
signal = SignalObj.create(x=np.array([1, 2, 3]),
                          y=np.array([10, 20, 30]))  # int arrays
# Internally converted to float64

# Validation: If you need strict float checks
from sigima.tools.checks import check_1d_array
check_1d_array(y)  # Allows float dtypes, raises for invalid types
```

### Working with ROI Boundaries

**Issue**: ROI extending beyond image causes `ValueError`.

**Solution**: Use `get_data()` which automatically clips:

```python
# Safe: Handles ROI clipping automatically
data = image.get_data(roi_index=0)

# Manual clipping (if needed in tools)
y0 = max(0, roi.y0)
y1 = min(image.data.shape[0], roi.y1)
x0 = max(0, roi.x0)
x1 = min(image.data.shape[1], roi.x1)
```

## Coding Conventions

### Type Annotations

```python
from __future__ import annotations

import numpy as np
from sigima.objects import SignalObj

def process(src: SignalObj, threshold: float) -> SignalObj:
    """Use forward references via __future__ import."""
    pass
```

### Docstrings

**Google-style** with Args/Returns:

```python
def my_function(x: np.ndarray, param: int) -> np.ndarray:
    """One-line summary.

    Longer description if needed.

    Args:
        x: Input array description
        param: Parameter description

    Returns:
        Output array description

    Raises:
        ValueError: When input is invalid
    """
```

For continued lines in enumerations (args, returns), indent subsequent lines by 1 space:

```python
def compute_feature(obj: SignalObj, param: MyParam) -> SignalObj:
    """Compute feature on signal.

    Args:
        obj: Input signal object
        param: Processing parameters, with a very long description that
         continues on the next line.

    Returns:
     Processed signal object
    """
```

### Imports

**Order**: Standard → Third-party → Sigima

```python
from __future__ import annotations

import numpy as np
import scipy.signal as sps
from guidata.dataset import DataSet

from sigima.objects import SignalObj
from sigima.proc.decorator import computation_function
from sigima.tools.checks import check_1d_array
```

### Module Exports

**Always define `__all__`**:

```python
__all__ = [
    "my_function",
    "MyParam",
    "AnotherFunction",
]
```

## Integration with DataLab

Sigima functions are **consumed by DataLab processors**:

1. **Sigima** implements computation: `sigima.proc.signal.my_feature()`
2. **DataLab** registers in processor: `self.register_1_to_1(sips.my_feature, ...)`
3. **DataLab** adds to menu: `self.processing_menu.addAction(act)`

**Testing flow**:
1. Test in **Sigima unit tests** (headless, fast)
2. Test in **DataLab integration tests** (GUI, full workflow)

## Key Files Reference

| File | Purpose |
|------|---------|
| `sigima/__init__.py` | Top-level exports (`SignalObj`, `ImageObj`, convenience functions) |
| `sigima/params.py` | Centralized parameter class exports (re-exports from `proc`) |
| `sigima/objects/signal/object.py` | `SignalObj` implementation |
| `sigima/objects/image/object.py` | `ImageObj` implementation |
| `sigima/proc/decorator.py` | `@computation_function` decorator system |
| `sigima/proc/signal/processing.py` | Signal processing functions (normalize, calibrate, etc.) |
| `sigima/proc/image/detection.py` | Image detection (blobs, peaks, contours) |
| `sigima/tools/checks.py` | Input validation (`check_1d_array`, `check_2d_array`) |
| `scripts/run_with_env.py` | Environment loader (always use for commands) |
| `.env` | Local PYTHONPATH for development |

## VS Code Tasks

`.vscode/tasks.json` provides shortcuts:

- **🧽🔦 Ruff**: Format + lint
- **🚀 Pytest**: Run tests (`--ff` flag)
- **📚 Compile translations**: Build .mo files
- **🔎 Scan translations**: Update .po files

## Release Classification

**Bug Fix** (1.0.x):
- Fixes incorrect behavior (e.g., ROI boundary clipping)
- Restores expected functionality (e.g., integer array conversion)
- Adds missing capability that should have existed (e.g., `replace_x_by_other_y`)

**Feature** (1.x.0):
- Entirely new computation type (e.g., new parametric images)
- New analysis methods

## Getting Help

- **Documentation**: https://sigima.readthedocs.io/
- **Issues**: https://github.com/DataLab-Platform/Sigima/issues
- **DataLab Integration**: https://datalab-platform.com/

---

**Remember**: Always use `scripts/run_with_env.py`, test headlessly with pytest, and export all parameters through `sigima.params`.
