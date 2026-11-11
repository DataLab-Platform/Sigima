.. _installation:

Installation
============

This section provides instructions on how to install and set up the Sigima project, including dependencies and environment configuration.

How to install
--------------

Sigima is available in several forms:

-   As a :ref:`install_conda`.

-   As a Python package, which can be installed using the :ref:`install_pip`.

-   As a precompiled :ref:`install_wheel`, which can be installed using ``pip``.

-   As a :ref:`install_source`, which can be installed using ``pip`` or manually.

.. seealso::

    Impatient to try the next version of Sigima? You can also install the
    latest development version of Sigima from the ``develop`` branch of the
    Git repository. See :ref:`install_development` for more information.

.. _install_conda:

Conda package
^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

To install ``sigima`` package from the `conda-forge` channel (https://anaconda.org/conda-forge/sigima), run the following command:

.. code-block:: console

    $ conda install conda-forge::sigima

.. _install_pip:

Package manager ``pip``
^^^^^^^^^^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

Sigima's package ``sigima`` is available on the Python Package Index (PyPI)
on the following URL: https://pypi.python.org/pypi/sigima.

Installing Sigima from PyPI is as simple as running this command
(you may need to use ``pip3`` instead of ``pip`` on some systems):

.. code-block:: console

    $ pip install sigima

.. note::

    If you already have a previous version of Sigima installed, you can
    upgrade it by running the same command with the ``--upgrade`` option:

    .. code-block:: console

        $ pip install --upgrade sigima

.. _install_optional:

Optional features
~~~~~~~~~~~~~~~~~

Some features of Sigima rely on optional dependencies, which may be installed
with the following extras:

-   ``opencv``: OpenCV (``opencv-python-headless``), required by the OpenCV blob
    detection (:func:`sigima.proc.image.blob_opencv`).

-   ``plotly``: Plotly, for the browser-based Plotly backend of :mod:`sigima.viz`.

.. code-block:: console

    $ pip install "sigima[opencv,plotly]"

Reading DICOM files requires the ``pydicom`` package, which is not installed by
any extra: install it separately (``pip install pydicom``). Without it, the DICOM
format is not available.

.. _install_wheel:

Wheel package
^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

On any operating system, using pip and the Wheel package is the easiest way to
install sigima on an existing Python distribution:

.. code-block:: console

    $ pip install --upgrade sigima-X.Y.Z-py3-none-any.whl

.. _install_source:

Source package
^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

Installing Sigima directly from the source package may be done using ``pip``:

.. code-block:: console

    $ pip install --upgrade sigima-X.Y.Z.tar.gz

Or, if you prefer, you can install it manually by running the following command
from the root directory of the source package:

.. code-block:: console

    $ pip install --upgrade .

Finally, you can also build your own Wheel package and install it using ``pip``,
by running the following command from the root directory of the source package
(this requires the ``build`` and ``wheel`` packages to be installed):

.. code-block:: console

    $ pip install build wheel  # Install build and wheel packages (if needed)
    $ python -m build  # Build the wheel package
    $ pip install --upgrade dist/sigima-X.Y.Z-py3-none-any.whl  # Install the wheel package

.. _install_development:

Development version
^^^^^^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

If you want to try the latest development version of Sigima, you can install
it directly from the ``develop`` branch of the Git repository (the default
branch, ``main``, holds the latest release).

The first time you install Sigima from the Git repository, enter the following
command:

.. code-block:: console

    $ pip install git+https://github.com/DataLab-Platform/Sigima.git@develop

Then, if at some point you want to upgrade to the latest version of Sigima,
just run the same command with options to force the reinstall of the package
without handling dependencies (because it would reinstall all dependencies):

.. code-block:: console

    $ pip install --force-reinstall --no-deps git+https://github.com/DataLab-Platform/Sigima.git@develop

.. note::

    If dependencies have changed, you may need to execute the same command as above,
    but without the ``--no-deps`` option.

Dependencies
------------

.. include:: ../requirements.rst
