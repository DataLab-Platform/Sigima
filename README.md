# Sigima - Scientific Image and Signal Processing Library

![Sigima](https://raw.githubusercontent.com/DataLab-Platform/Sigima/main/doc/images/Sigima-Banner.svg)

[![license](https://img.shields.io/pypi/l/sigima.svg)](./LICENSE)
[![pypi version](https://img.shields.io/pypi/v/sigima.svg)](https://pypi.org/project/sigima/)
[![PyPI status](https://img.shields.io/pypi/status/sigima.svg)](https://github.com/DataLab-Platform/Sigima)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/sigima.svg)](https://pypi.org/project/sigima/)

**Sigima** is an **open-source Python library for scientific image and signal processing**,
designed as a modular and testable foundation for building advanced analysis pipelines.

🔬 Developed by [Codra](https://codra.net/) and the [DataLab Platform Developers](https://github.com/DataLab-Platform), Sigima powers the computation backend of [DataLab](https://datalab-platform.com/).

---

## ✨ Highlights

- Unified processing model for **1D signals** and **2D images**
- Works with **object-oriented wrappers** (`SignalObj`, `ImageObj`) extending NumPy arrays
- Includes common algorithms: filtering, smoothing, binning, thresholding, labeling, etc.
- Structured for **testability**, **modularity**, and **headless usage**
- 100% **independent of GUI frameworks** (no Qt/PlotPyStack dependencies)

---

## 💡 Use cases

Sigima is meant to be:

- A **processing backend** for scientific/industrial tools
- A library to **build reproducible analysis pipelines**
- A component for **headless automation or remote execution**
- A testbed for **developing and validating new signal/image operations**

---

## 📖 Design Philosophy

The main goal of **Sigima** is to provide a unified, high-level API for handling and processing **1D signals** and **2D images**, through dedicated Python objects: `SignalObj` and `ImageObj`.

The library is organized to separate concerns clearly:

- `sigima.obj`: defines the object model for signals and images.
- `sigima.computation`: provides processing functions that operate directly on these objects.
- `sigima.io`: handles input/output operations (CSV, images, etc.) for signal/image objects.
- `sigima.algorithms`: contains low-level helper functions that complement NumPy, SciPy or scikit-image — these are **internal tools**, not part of the public API.

This structure allows users and developers to write readable, object-oriented signal/image processing code, while keeping a clean separation between user-facing APIs and internal building blocks.

> 📌 **Note:** `sigima.algorithms` is not meant to be used directly. Its purpose is to support the implementation of higher-level functions in `sigima.computation`.

---

## 📦 Installation

```bash
pip install sigima
```

Or in a development environment:

```bash
git clone https://github.com/DataLab-Platform/Sigima.git
cd Sigima
pip install -e .
```

---

## 📚 Documentation

📖 Full documentation (in progress) is available at:
👉 <https://datalab-platform.com/en/api/>

> Want to use Sigima inside DataLab with GUI tools?
> Check out the full platform: [DataLab](https://datalab-platform.com/)

---

## ⚙️ Architecture

Sigima is organized by data type:

```text
sigima/
├── algorithms/   # ❌ Internal low-level algorithms (not part of public API)
├── computation/  # ✅ High-level processing functions operating on SignalObj/ImageObj
│   ├── base/     #    Common processing functions
│   ├── signal/   #    1D signal processing
│   └── image/    #    2D image processing
```

Each domain provides:

- Low-level functions operating on NumPy arrays
- High-level functions operating on `SignalObj` or `ImageObj`

---

## 🧪 Testing

Sigima comes with unit tests based on `pytest`.

To run all tests:

```bash
pytest
```

To run GUI-assisted validation tests (optional):

```bash
pytest --gui
```

---

## 🧠 License

Sigima is distributed under the terms of the BSD 3-Clause license.
See [LICENSE](./LICENSE) for details.

---

## 🤝 Contributing

Bug reports, feature requests and pull requests are welcome!
See the [CONTRIBUTING](https://datalab-platform.com/en/contributing) guide to get started.

---

![Python](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/Python.png)
![NumPy](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/NumPy.png)
![SciPy](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/SciPy.png)
![scikit-image](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/scikit-image.png)
![OpenCV](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/OpenCV.png)

---

© Codra / DataLab Platform Developers
