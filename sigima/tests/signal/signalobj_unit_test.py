# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests around the `SignalObj` class and its creation from parameters.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import os.path as osp

import numpy as np
import pytest

import sigima.io
import sigima.objects
from sigima.io.signal import SignalIORegistry
from sigima.tests import guiutils
from sigima.tests.data import iterate_signal_creation
from sigima.tests.env import execenv
from sigima.tests.helpers import (
    WorkdirRestoringTempDir,
    compare_metadata,
    read_test_objects,
)


# pylint: disable=unused-argument
def preprocess_signal_parameters(param: sigima.objects.NewSignalParam) -> None:
    """Preprocess signal parameters before creating the signal.

    Args:
        param: The signal parameters to preprocess.
    """
    # Add here specific preprocessing for signal parameters if needed


def postprocess_signal_object(
    obj: sigima.objects.SignalObj, stype: sigima.objects.SignalTypes
) -> None:
    """Postprocess signal object after creation.

    Args:
        obj: The signal object to postprocess.
        stype: The type of the signal.
    """
    if stype == sigima.objects.SignalTypes.ZERO:
        assert (obj.y == 0).all()


def test_all_signal_types() -> None:
    """Test all combinations of signal types and data sizes"""
    execenv.print(f"{test_all_signal_types.__doc__}:")
    for signal in iterate_signal_creation(
        preproc=preprocess_signal_parameters, postproc=postprocess_signal_object
    ):
        assert signal.x is not None and signal.y is not None
    execenv.print(f"{test_all_signal_types.__doc__}: OK")


@pytest.mark.parametrize(
    "fname, orig_signal", list(read_test_objects(SignalIORegistry))
)
def test_hdf5_signal_io(fname: str, orig_signal: sigima.objects.SignalObj) -> None:
    """Test HDF5 I/O for signal objects"""
    if orig_signal is None:
        pytest.skip(f"Skipping {fname} (not implemented)")
    execenv.print(f"{test_hdf5_signal_io.__doc__}:")
    with WorkdirRestoringTempDir() as tmpdir:
        # Save to HDF5
        filename = osp.join(tmpdir, f"test_{osp.basename(fname)}.h5sig")
        sigima.io.write_signal(filename, orig_signal)
        execenv.print(f"  Saved {filename}")
        # Read back
        fetch_signal = sigima.io.read_signal(filename)
        execenv.print(f"  Read {filename}")
        orig_x, orig_y = orig_signal.x, orig_signal.y
        orig_x: np.ndarray
        orig_y: np.ndarray
        x, y = fetch_signal.x, fetch_signal.y
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape == orig_x.shape
        assert y.shape == orig_y.shape
        assert x.dtype == orig_x.dtype
        assert y.dtype == orig_y.dtype
        assert np.isclose(x, orig_x, atol=0.0).all()
        assert np.isclose(y, orig_y, atol=0.0).all()
        try:
            compare_metadata(
                fetch_signal.metadata, orig_signal.metadata.copy(), raise_on_diff=True
            )
        except AssertionError as exc:
            raise AssertionError(
                f"Signal metadata read from file does not match original ({fname})"
            ) from exc
    execenv.print(f"{test_hdf5_signal_io.__doc__}: OK")


@pytest.mark.gui
def test_signal_parameters_interactive() -> None:
    """Test interactive creation of signal parameters"""
    execenv.print(f"{test_signal_parameters_interactive.__doc__}:")
    with guiutils.lazy_qt_app_context(force=True):
        for stype in sigima.objects.SignalTypes:
            param = sigima.objects.create_signal_parameters(stype)
            if isinstance(param, sigima.objects.CustomSignalParam):
                param.setup_array()
            if param.edit():
                execenv.print(f"  Edited parameters for {stype.value}:")
                execenv.print(f"    {param}")
            else:
                execenv.print(f"  Skipped editing parameters for {stype.value}")
    execenv.print(f"{test_signal_parameters_interactive.__doc__}: OK")


def test_create_signal() -> None:
    """Test creation of a signal object using `create_signal` function"""
    execenv.print(f"{test_create_signal.__doc__}:")
    # pylint: disable=import-outside-toplevel

    # Test all combinations of input parameters
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    dx = np.full_like(x, 0.1)
    dy = np.full_like(y, 0.01)
    metadata = {"source": "test", "description": "Test signal"}
    units = ("s", "V")
    labels = ("Time", "Amplitude")

    # 1. Create signal with all parameters
    title = "Some Signal"
    signal = sigima.objects.create_signal(
        title=title,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        metadata=metadata,
        units=units,
        labels=labels,
    )
    assert isinstance(signal, sigima.objects.SignalObj)
    assert signal.title == title
    assert np.array_equal(signal.x, x)
    assert np.array_equal(signal.y, y)
    assert np.array_equal(signal.dx, dx)
    assert np.array_equal(signal.dy, dy)
    assert signal.metadata == metadata
    assert (signal.xunit, signal.yunit) == units
    assert (signal.xlabel, signal.ylabel) == labels

    # 2. Create signal with only x and y
    signal = sigima.objects.create_signal("", x=x, y=y)
    assert isinstance(signal, sigima.objects.SignalObj)
    assert np.array_equal(signal.x, x)
    assert np.array_equal(signal.y, y)
    assert signal.dx is None
    assert signal.dy is None
    assert not signal.metadata
    assert (signal.xunit, signal.yunit) == ("", "")
    assert (signal.xlabel, signal.ylabel) == ("", "")

    # 3. Create signal with only x, y, and dx
    signal = sigima.objects.create_signal("", x=x, y=y, dx=dx)
    assert isinstance(signal, sigima.objects.SignalObj)
    assert np.array_equal(signal.x, x)
    assert np.array_equal(signal.y, y)
    assert np.array_equal(signal.dx, dx)
    assert signal.dy is None

    # 4. Create signal with only x, y, and dy
    signal = sigima.objects.create_signal("", x=x, y=y, dy=dy)
    assert isinstance(signal, sigima.objects.SignalObj)
    assert np.array_equal(signal.x, x)
    assert np.array_equal(signal.y, y)
    assert signal.dx is None
    assert np.array_equal(signal.dy, dy)

    execenv.print(f"{test_create_signal.__doc__}: OK")


def test_create_signal_from_param() -> None:
    """Test creation of a signal object using `create_signal_from_param` function"""
    execenv.print(f"{test_create_signal_from_param.__doc__}:")

    # Test with different signal parameter types
    test_cases = [
        # Basic periodic functions
        (sigima.objects.SineParam, "sine"),
        (sigima.objects.CosineParam, "cosine"),
        (sigima.objects.SawtoothParam, "sawtooth"),
        (sigima.objects.TriangleParam, "triangle"),
        (sigima.objects.SquareParam, "square"),
        (sigima.objects.SincParam, "sinc"),
        # Mathematical functions
        (sigima.objects.GaussParam, "gaussian"),
        (sigima.objects.LorentzParam, "lorentzian"),
        (sigima.objects.ExponentialParam, "exponential"),
        (sigima.objects.LogisticParam, "logistic"),
        (sigima.objects.LinearChirpParam, "linear_chirp"),
        (sigima.objects.StepParam, "step"),
        (sigima.objects.PulseParam, "pulse"),
        (sigima.objects.SquarePulseParam, "square_pulse"),
        (sigima.objects.StepPulseParam, "step_pulse"),
        (sigima.objects.PolyParam, "polynomial"),
        # Noise and random signals
        (sigima.objects.NormalDistribution1DParam, "normal_noise"),
        (sigima.objects.PoissonDistribution1DParam, "poisson_noise"),
        (sigima.objects.UniformDistribution1DParam, "uniform_noise"),
        (sigima.objects.ZeroParam, "zero"),
        # Other signals
        (sigima.objects.CustomSignalParam, "custom"),
        (sigima.objects.VoigtParam, "voigt"),
        (sigima.objects.PlanckParam, "planck"),
    ]

    # Raise an exception if sigima.objects.signal contain *Param classes not listed here
    param_classes = dict(test_cases)
    for attr_name in dir(sigima.objects):
        attr = getattr(sigima.objects, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, sigima.objects.NewSignalParam)
            and attr is not sigima.objects.NewSignalParam
            and attr is not sigima.objects.CustomSignalParam
            and attr not in param_classes
        ):
            raise AssertionError(f"Missing test case for {attr.__name__}")

    for param_class, name in test_cases:
        # Create parameter instance with default values
        param = param_class.create(size=100, xmin=1.0, xmax=10.0)
        param.title = f"Test {name} signal"

        # Test the function
        signal = sigima.objects.create_signal_from_param(param)

        # Verify the returned object
        assert isinstance(signal, sigima.objects.SignalObj), (
            f"Expected SignalObj, got {type(signal)} for {name}"
        )
        assert signal.title == f"Test {name} signal", (
            f"Title mismatch for {name}: expected 'Test {name} signal', "
            f"got '{signal.title}'"
        )
        assert signal.x is not None, f"X data is None for {name}"
        assert signal.y is not None, f"Y data is None for {name}"
        assert len(signal.x) == 100, f"X length mismatch for {name}"
        assert len(signal.y) == 100, f"Y length mismatch for {name}"
        assert isinstance(signal.x, np.ndarray), f"X is not ndarray for {name}"
        assert isinstance(signal.y, np.ndarray), f"Y is not ndarray for {name}"

        # Test automatic title generation for parameters that support it
        param_autotitle = param_class.create(size=100, xmin=1.0, xmax=10.0)
        param_autotitle.title = ""  # Empty title to trigger auto-generation
        signal_autotitle = sigima.objects.create_signal_from_param(param_autotitle)
        # Distribution params should generate descriptive titles
        if "Distribution" in param_class.__name__:
            assert signal_autotitle.title != "", (
                f"Title should be auto-generated for {name}"
            )
            assert "Random" in signal_autotitle.title, (
                f"Auto-generated title should contain 'Random' for {name}"
            )

        execenv.print(f"  Created {name} signal: OK")

    # Test with custom parameters and title generation
    param = sigima.objects.GaussParam.create(size=50, xmin=-5.0, xmax=5.0)
    param.title = ""  # Empty title should trigger automatic numbering
    signal = sigima.objects.create_signal_from_param(param)

    assert signal.title != "", "Empty title should be replaced"

    # Test parameter validation with units and labels
    param = sigima.objects.SineParam()
    param.title = "Sine wave test"
    # xunit is set by default to "s" in SineParam
    assert param.xunit == "s"
    param.yunit = "V"
    param.xlabel = "Time"
    param.ylabel = "Amplitude"

    signal = sigima.objects.create_signal_from_param(param)

    expected_xunit = "s"
    assert signal.xunit == expected_xunit, (
        f"X unit mismatch: expected '{expected_xunit}', got '{signal.xunit}'"
    )
    expected_yunit = "V"
    assert signal.yunit == expected_yunit, (
        f"Y unit mismatch: expected '{expected_yunit}', got '{signal.yunit}'"
    )
    expected_xlabel = "Time"
    assert signal.xlabel == expected_xlabel, (
        f"X label mismatch: expected '{expected_xlabel}', got '{signal.xlabel}'"
    )
    expected_ylabel = "Amplitude"
    assert signal.ylabel == expected_ylabel, (
        f"Y label mismatch: expected '{expected_ylabel}', got '{signal.ylabel}'"
    )

    execenv.print(f"{test_create_signal_from_param.__doc__}: OK")


def test_custom_signal_preserves_edited_array() -> None:
    """Test that editing CustomSignalParam.xyarray is preserved on regeneration"""
    execenv.print(f"{test_custom_signal_preserves_edited_array.__doc__}:")

    param = sigima.objects.CustomSignalParam.create(size=5, xmin=0.0, xmax=1.0)

    # First generation initializes the array from size/xmin/xmax.
    x0, y0 = param.generate_1d_data()
    assert np.allclose(x0, np.linspace(0.0, 1.0, 5))
    assert np.allclose(y0, x0)

    # Simulate a user editing the XY values in the array editor.
    edited = np.array([[0.0, 10.0], [0.25, 20.0], [0.5, 30.0], [0.75, 40.0]])
    param.xyarray = edited

    # Regenerating must keep the edited values instead of clobbering them.
    x1, y1 = param.generate_1d_data()
    assert np.allclose(x1, edited[:, 0]), "Edited X values were not preserved"
    assert np.allclose(y1, edited[:, 1]), "Edited Y values were not preserved"

    execenv.print(f"{test_custom_signal_preserves_edited_array.__doc__}: OK")


def test_signal_copy() -> None:
    """Test copying signal objects with all attributes"""
    execenv.print(f"{test_signal_copy.__doc__}:")

    # Create a base signal with some data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    dx = np.full_like(x, 0.1)
    dy = np.full_like(y, 0.01)
    title = "Original Signal"
    metadata = {"key1": "value1", "key2": 42}
    units = ("s", "V")
    labels = ("Time", "Voltage")

    # Test 1: Copy signal with all attributes
    execenv.print("  Test 1: Copy signal with all attributes")
    signal = sigima.objects.create_signal(
        title=title,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        metadata=metadata.copy(),
        units=units,
        labels=labels,
    )

    # Set scale attributes
    signal.autoscale = False
    signal.xscalelog = True
    signal.xscalemin = 1.0
    signal.xscalemax = 9.0
    signal.yscalelog = False
    signal.yscalemin = -1.5
    signal.yscalemax = 1.5

    # Copy the signal
    copied = signal.copy()

    # Verify the copy
    assert copied is not signal
    assert copied.title == signal.title
    assert np.array_equal(copied.x, signal.x)
    assert np.array_equal(copied.y, signal.y)
    assert np.array_equal(copied.dx, signal.dx)
    assert np.array_equal(copied.dy, signal.dy)
    assert copied.xydata is not signal.xydata  # Different array objects
    assert copied.metadata == signal.metadata
    assert copied.metadata is not signal.metadata
    assert (copied.xunit, copied.yunit) == units
    assert (copied.xlabel, copied.ylabel) == labels

    # Verify scale attributes are preserved
    assert copied.autoscale == signal.autoscale
    assert copied.xscalelog == signal.xscalelog
    assert copied.xscalemin == signal.xscalemin
    assert copied.xscalemax == signal.xscalemax
    assert copied.yscalelog == signal.yscalelog
    assert copied.yscalemin == signal.yscalemin
    assert copied.yscalemax == signal.yscalemax
    execenv.print("    ✓ All attributes correctly copied")

    # Test 2: Copy with title override
    execenv.print("  Test 2: Copy with custom title")
    new_title = "Copied Signal"
    copied_with_title = signal.copy(title=new_title)
    assert copied_with_title.title == new_title
    assert copied_with_title.autoscale == signal.autoscale
    assert np.array_equal(copied_with_title.x, signal.x)
    execenv.print("    ✓ Title override works correctly")

    # Test 3: Copy with metadata filtering
    execenv.print("  Test 3: Copy with metadata filtering")
    copied_basic_meta = signal.copy(all_metadata=False)
    assert copied_basic_meta.autoscale == signal.autoscale
    assert copied_basic_meta.xscalelog == signal.xscalelog
    execenv.print("    ✓ Metadata filtering works correctly")

    # Test 4: Copy signal without error bars
    execenv.print("  Test 4: Copy signal without error bars")
    signal_no_err = sigima.objects.create_signal(
        title="Signal without error bars",
        x=x,
        y=y,
        units=units,
        labels=labels,
    )
    signal_no_err.autoscale = True
    signal_no_err.yscalelog = True

    copied_no_err = signal_no_err.copy()
    assert copied_no_err.dx is None
    assert copied_no_err.dy is None
    assert copied_no_err.autoscale is True
    assert copied_no_err.yscalelog is True
    execenv.print("    ✓ Signal without error bars copied correctly")

    execenv.print(f"{test_signal_copy.__doc__}: OK")


def test_coordinate_conversion() -> None:
    """Test physical_to_indices and indices_to_physical methods"""
    execenv.print(f"{test_coordinate_conversion.__doc__}:")

    # Create test signals with different x-coordinate patterns
    n = 100

    # ==================== Test 1: Uniform spacing ====================
    execenv.print("  Test 1: Uniform spacing - basic conversion")
    x_uniform = np.linspace(0.0, 10.0, n)
    y_uniform = np.sin(x_uniform)
    signal_uniform = sigima.objects.create_signal(
        title="Uniform Spacing Test", x=x_uniform, y=y_uniform
    )

    # Test forward conversion (physical → indices)
    # Since SignalObj uses argmin to find closest x, we test with exact x values
    test_coords = [0.0, 5.0, 10.0]
    indices = signal_uniform.physical_to_indices(test_coords)
    assert len(indices) == 3
    assert indices[0] == 0  # Closest to x[0] = 0.0
    assert indices[1] == 49  # Closest to x[49] ≈ 5.0 (for n=100, linspace 0-10)
    assert indices[2] == 99  # Closest to x[99] = 10.0
    execenv.print("    ✓ Forward conversion (physical → indices) correct")

    # Test backward conversion (indices → physical)
    test_indices = [0, 49, 99]
    coords = signal_uniform.indices_to_physical(test_indices)
    assert len(coords) == 3
    np.testing.assert_allclose(coords[0], 0.0, rtol=1e-10)
    np.testing.assert_allclose(coords[1], 5.0, rtol=0.02)  # ~1% tolerance
    np.testing.assert_allclose(coords[2], 10.0, rtol=1e-10)
    execenv.print("    ✓ Backward conversion (indices → physical) correct")

    # Test round-trip accuracy
    execenv.print("  Test 2: Uniform spacing - round-trip accuracy")
    # Use exact x values for perfect round-trip
    original_coords = [x_uniform[10], x_uniform[50], x_uniform[80]]
    indices_rt = signal_uniform.physical_to_indices(original_coords)
    recovered_coords = signal_uniform.indices_to_physical(indices_rt)
    np.testing.assert_allclose(recovered_coords, original_coords, rtol=1e-10)
    execenv.print("    ✓ Round-trip (physical → indices → physical) preserves values")

    # ==================== Test 3: Non-uniform spacing ====================
    execenv.print("  Test 3: Non-uniform spacing - logarithmic")
    x_log = np.logspace(0, 2, n)  # 1 to 100, logarithmic spacing
    y_log = np.sin(x_log)
    signal_log = sigima.objects.create_signal(
        title="Logarithmic Spacing Test", x=x_log, y=y_log
    )

    # Test with exact x values
    test_coords_log = [x_log[0], x_log[50], x_log[99]]
    indices_log = signal_log.physical_to_indices(test_coords_log)
    assert indices_log[0] == 0
    assert indices_log[1] == 50
    assert indices_log[2] == 99
    execenv.print("    ✓ Non-uniform forward conversion correct")

    # Test backward conversion
    coords_log = signal_log.indices_to_physical([0, 50, 99])
    np.testing.assert_allclose(coords_log[0], x_log[0], rtol=1e-10)
    np.testing.assert_allclose(coords_log[1], x_log[50], rtol=1e-10)
    np.testing.assert_allclose(coords_log[2], x_log[99], rtol=1e-10)
    execenv.print("    ✓ Non-uniform backward conversion correct")

    # ==================== Test 4: Finding closest value ====================
    execenv.print("  Test 4: Finding closest value (argmin behavior)")
    # Test that physical_to_indices finds the closest x value
    # For uniform spacing, test a value between grid points
    test_val = 5.05  # Between x[49] and x[50]
    idx = signal_uniform.physical_to_indices([test_val])
    # Should return index of closest value
    expected_idx = np.abs(x_uniform - test_val).argmin()
    assert idx[0] == expected_idx
    execenv.print("    ✓ Finds closest x value correctly (argmin)")

    # Test with multiple values not on grid
    test_vals = [1.23, 4.56, 7.89]
    indices_approx = signal_uniform.physical_to_indices(test_vals)
    for i, val in enumerate(test_vals):
        expected = np.abs(x_uniform - val).argmin()
        assert indices_approx[i] == expected
    execenv.print("    ✓ Multiple approximate values handled correctly")

    # ==================== Test 5: Quadratic spacing ====================
    execenv.print("  Test 5: Non-uniform spacing - quadratic")
    x_quad = np.linspace(0, 1, n) ** 2 * 100  # Quadratic spacing, denser near 0
    y_quad = np.exp(-x_quad / 10)
    signal_quad = sigima.objects.create_signal(
        title="Quadratic Spacing Test", x=x_quad, y=y_quad
    )

    # Round-trip test with exact values
    test_indices_quad = [0, 25, 50, 75, 99]
    coords_quad = signal_quad.indices_to_physical(test_indices_quad)
    indices_back = signal_quad.physical_to_indices(coords_quad)
    assert indices_back == test_indices_quad
    execenv.print("    ✓ Round-trip for quadratic spacing preserves indices")

    # ==================== Test 6: Edge cases ====================
    execenv.print("  Test 6: Edge cases")

    # Empty coordinate list
    empty_coords = []
    empty_indices = signal_uniform.physical_to_indices(empty_coords)
    assert len(empty_indices) == 0
    execenv.print("    ✓ Empty coordinate list handled")

    # Single point
    single_coord = [5.0]
    single_idx = signal_uniform.physical_to_indices(single_coord)
    assert len(single_idx) == 1
    execenv.print("    ✓ Single point conversion works")

    # Multiple points
    multi_coords = [0.0, 2.5, 5.0, 7.5, 10.0]
    multi_idx = signal_uniform.physical_to_indices(multi_coords)
    assert len(multi_idx) == 5
    execenv.print("    ✓ Multiple points conversion works")

    # Boundary values
    boundary_coords = [x_uniform[0], x_uniform[-1]]
    boundary_idx = signal_uniform.physical_to_indices(boundary_coords)
    assert boundary_idx[0] == 0
    assert boundary_idx[1] == n - 1
    execenv.print("    ✓ Boundary values handled correctly")

    # ==================== Test 7: Out-of-range values ====================
    execenv.print("  Test 7: Out-of-range values")
    # Values outside the x range should map to closest endpoint
    out_of_range = [-100.0, 200.0]
    out_idx = signal_uniform.physical_to_indices(out_of_range)
    assert out_idx[0] == 0  # Closest to minimum x
    assert out_idx[1] == n - 1  # Closest to maximum x
    execenv.print("    ✓ Out-of-range values map to closest endpoint")

    # ==================== Test 8: Complex y data ====================
    execenv.print("  Test 8: Complex y data")
    # SignalObj can have complex y values, but x is always real
    x_complex = np.linspace(0, 2 * np.pi, n)
    y_complex = np.exp(1j * x_complex)  # Complex exponential
    signal_complex = sigima.objects.create_signal(
        title="Complex Signal Test", x=x_complex, y=y_complex
    )

    # Test that coordinate conversion still works with complex y
    test_coords_complex = [0.0, np.pi, 2 * np.pi]
    indices_complex = signal_complex.physical_to_indices(test_coords_complex)
    assert len(indices_complex) == 3
    # Verify we can recover coordinates
    coords_complex = signal_complex.indices_to_physical(indices_complex)
    np.testing.assert_allclose(coords_complex, test_coords_complex, rtol=0.02)
    execenv.print("    ✓ Complex y data doesn't affect coordinate conversion")

    # ==================== Test 9: Dense data near specific region ====================
    execenv.print("  Test 9: Non-uniform spacing - dense region")
    # Create a signal with very dense sampling in middle region
    x_dense = np.concatenate(
        [
            np.linspace(0, 1, 10),  # Sparse
            np.linspace(1, 2, 70),  # Dense
            np.linspace(2, 3, 10),  # Sparse
        ]
    )
    y_dense = np.sin(x_dense * 2 * np.pi)
    signal_dense = sigima.objects.create_signal(
        title="Dense Region Test", x=x_dense, y=y_dense
    )

    # Test conversion in dense region
    dense_coords = [1.5]  # Middle of dense region
    dense_idx = signal_dense.physical_to_indices(dense_coords)
    recovered = signal_dense.indices_to_physical(dense_idx)
    # Should find a very close match due to dense sampling
    assert abs(recovered[0] - 1.5) < 0.02
    execenv.print("    ✓ Dense sampling region handled correctly")

    execenv.print(f"{test_coordinate_conversion.__doc__}: OK")


def test_set_dy_does_not_create_phantom_dx() -> None:
    """Setting only ``dy`` must leave ``dx`` as ``None`` (not phantom zeros).

    Regression test: ``__set_dy`` used to extend ``xydata`` with ``np.zeros``,
    so ``dx`` was silently reported as an array of zeros instead of ``None``.
    """
    x = np.linspace(0.0, 1.0, 5)
    obj = sigima.objects.create_signal("test", x, x.copy())
    assert obj.dx is None
    assert obj.dy is None

    obj.dy = 0.1 * np.ones_like(x)

    assert obj.dy is not None
    assert obj.dx is None, (
        f"Setting dy should not create phantom dx values, got {obj.dx!r}"
    )


def test_set_dx_does_not_create_phantom_dy() -> None:
    """Setting only ``dx`` must leave ``dy`` as ``None`` (not phantom zeros)."""
    x = np.linspace(0.0, 1.0, 5)
    obj = sigima.objects.create_signal("test", x, x.copy())
    assert obj.dx is None
    assert obj.dy is None

    obj.dx = 0.01 * np.ones_like(x)

    assert obj.dx is not None
    assert obj.dy is None, (
        f"Setting dx should not create phantom dy values, got {obj.dy!r}"
    )


def run_all_tests() -> None:
    """Run all tests in this module"""
    test_signal_parameters_interactive()
    test_all_signal_types()
    for fname, orig_signal in read_test_objects(SignalIORegistry):
        test_hdf5_signal_io(fname, orig_signal)
    test_create_signal()
    test_create_signal_from_param()
    test_signal_copy()
    test_coordinate_conversion()
    test_set_dy_does_not_create_phantom_dx()
    test_set_dx_does_not_create_phantom_dy()


if __name__ == "__main__":
    run_all_tests()
