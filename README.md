# Sigima - Scientific Image and Signal Processing Library

![Sigima](https://raw.githubusercontent.com/DataLab-Platform/Sigima/main/doc/images/Sigima-Banner.svg)

[![license](https://img.shields.io/pypi/l/sigima.svg)](./LICENSE)
[![pypi version](https://img.shields.io/pypi/v/sigima.svg)](https://pypi.org/project/sigima/)
[![PyPI status](https://img.shields.io/pypi/status/sigima.svg)](https://github.com/DataLab-Platform/Sigima)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/sigima.svg)](https://pypi.org/project/sigima/)

**Sigima** is an **open-source Python library for scientific image and signal processing**,
designed as a modular and testable foundation for building advanced analysis pipelines.

🔬 Developed by the [DataLab Platform Developers](https://github.com/DataLab-Platform), Sigima powers the computation backend of [DataLab](https://datalab-platform.com/).

## 🌟 Project & Sponsors

| Project/Sponsor     | Description |
|---------------------|-------------|
| <a href="https://datalab-platform.com/"><img src="https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/resources/DataLab-Banner.svg" alt="DataLab logo" style="height:80px;"/></a> | Open-source platform for scientific signal and image processing, powered by Sigima. |
| <a href="https://nlnet.nl/"><img src="https://nlnet.nl/logo/banner.svg" alt="NLnet logo" style="height:80px;width:209px;"/></a> | European non-profit supporting open-source and internet projects. Sigima has received funding from NLnet for its development, through the DataLab project. |

---

## ✨ Highlights

- Unified processing model for **1D signals** and **2D images**
- Works with **object-oriented wrappers** (`SignalObj`, `ImageObj`) extending NumPy arrays
- Includes common processing tasks: filtering, smoothing, binning, thresholding, labeling, etc.
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

- `sigima.objects`: defines the object model for signals and images.
- `sigima.proc`: provides high-level processing functions that operate directly on `SignalObj` and `ImageObj` instances.
- `sigima.io`: handles input/output operations (CSV files, image formats, etc.) for signals and images.
- `sigima.tools`: contains **low-level, NumPy-based functions** that implement the core logic behind many processing routines.

This structure supports a **layered programming model**:

- Developers can use `computation` to process full signal/image objects in an object-oriented manner.
- Or they can directly use `tools` to process raw NumPy arrays — for instance, in custom tools or when integrating Sigima into other projects.

> ⚠️ `sigima.tools` is not intended as a general-purpose NumPy extension. Its purpose is to **fill in the gaps** of common scientific libraries (NumPy, SciPy, scikit-image, etc.), offering consistent tools for signal/image processing in the context of Sigima and similar projects.

---

## Usage Outside Sigima

Although Sigima is designed primarily for object-based processing, some of its core functions are useful on their own.

For instance, the [DataLab](https://datalab-platform.com) project — an open-source platform for signal/image processing — uses many functions from `sigima.tools` independently of the object model. This demonstrates how `sigima.tools` can serve as a **lightweight utility layer** in scientific and industrial Python applications, even when the object model is not used directly.

To maintain this flexibility and avoid confusion, the distinction between `tools` (array-based) and `computation` (object-based) is intentional and explicit.

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
├── tools/      # Low-level NumPy-based algorithms supporting some computation functions
├── proc/       # High-level processing functions operating on SignalObj/ImageObj
│   ├── base/   # Common processing functions
│   ├── signal/ # 1D signal processing
│   └── image/  # 2D image processing
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

© DataLab Platform Developers
