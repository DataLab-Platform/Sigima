# Changelog #

The `sigima` library is part of the DataLab open-source platform.
See DataLab [roadmap page](https://datalab-platform.com/en/contributing/roadmap.html) for future and past milestones.

## sigima 0.3.0 ##

💥 New features and enhancements:

* New image processing features:
  * Added new "Gaussian frequency filter" feature:
    * This feature allows to filter an image in the frequency domain using a Gaussian filter.
    * It is implemented in the `sigima.proc.image.freq_fft` function.

* New signal processing features:
  * Added "Brickwall frequency filter" feature:
    * This feature allows to filter a signal in the frequency domain using a brickwall filter.
    * It is implemented in the `sigima.proc.signal.freq_fft` function, among the other frequency domain filtering features that were already available (e.g., `Bessel`, `Butterworth`, etc.).

## sigima 0.2.0 ##

⚠️ Major API changes:

* Rename `sigima.computation` to `sigima.proc`.
* Rename `sigima.algorithms` to `sigima.tools`.
* Rename `sigima.obj` to `sigima.objects`.
* Rename `sigima.param` to `sigima.params`.

ℹ️ Various changes:

* Add Sigima SVG logo.

🛠️ Bug fixes:

* Fix API documentation and docstrings.

## sigima 0.1.0 ##

This first version of the library is the result of the externalization of the signal and image processing features from the DataLab main repository.
