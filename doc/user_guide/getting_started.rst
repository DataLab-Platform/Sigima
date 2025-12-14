.. _getting_started:

Getting Started
===============

This page provides a quick introduction to get you started with Sigima through practical examples.

Interactive Notebook
--------------------

The best way to start learning Sigima is through an interactive Jupyter notebook. This example demonstrates image processing with Regions of Interest (ROI) and uses the matplotlib backend for web-friendly visualization.

:download:`Download simple_example.ipynb <../simple_example.ipynb>`

.. toctree::
   :maxdepth: 1

   ../simple_example

What This Example Shows
^^^^^^^^^^^^^^^^^^^^^^^^

The notebook demonstrates:

* Creating image objects from NumPy arrays
* Defining Regions of Interest (ROI) for selective processing
* Applying Gaussian filtering to images
* Visualizing results with matplotlib backend (web-friendly, no Qt required)
* Using the new visualization backend selection feature introduced in Sigima V1.1

Key Concepts
^^^^^^^^^^^^

**Object-Oriented Approach**
   Sigima wraps NumPy arrays in rich objects (``SignalObj``, ``ImageObj``) that carry metadata, units, and ROI information.

**Region of Interest (ROI)**
   Process only specific areas of your data by defining ROIs. The example shows a circular ROI applied to an image.

**Backend Flexibility**
   Choose between PlotPy (Qt-based, interactive) or Matplotlib (web-friendly) backends for visualization, making Sigima suitable for both desktop applications and web-based workflows.

Remote Control Example
----------------------

Sigima also enables remote control of DataLab sessions from external scripts or notebooks. This is useful for automation, batch processing, or integrating DataLab into larger workflows.

:download:`Download remote_example.ipynb <../remote_example.ipynb>`

.. toctree::
   :maxdepth: 1

   ../remote_example

This notebook demonstrates:

* Connecting to a running DataLab instance using ``SimpleRemoteProxy``
* Adding signals and images to DataLab from external Python code
* Controlling DataLab programmatically without GUI interaction

Next Steps
----------

* Explore the :ref:`features` page for a complete overview of available operations
* Check out the :ref:`API documentation <api>` for detailed function references
* Browse the :doc:`gallery of examples </auto_examples/index>` for more use cases
