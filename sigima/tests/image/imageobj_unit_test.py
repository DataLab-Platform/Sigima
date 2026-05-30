# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests around the `ImageObj` class and its creation from parameters.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import os.path as osp

import numpy as np
import pytest

import sigima.io
import sigima.objects
from sigima.io.image import ImageIORegistry
from sigima.objects.image import (
    Checkerboard2DParam,
    Gauss2DParam,
    Ramp2DParam,
    Ring2DParam,
    SiemensStar2DParam,
    Sinc2DParam,
    SinusoidalGrating2DParam,
)
from sigima.tests import guiutils
from sigima.tests.data import (
    create_annotated_image,
    create_test_image_with_metadata,
    iterate_image_creation,
)
from sigima.tests.env import execenv
from sigima.tests.helpers import (
    WorkdirRestoringTempDir,
    check_scalar_result,
    compare_metadata,
    read_test_objects,
)


def preprocess_image_parameters(param: sigima.objects.NewImageParam) -> None:
    """Preprocess image parameters before creating the image.

    Args:
        param: The image parameters to preprocess.
    """
    if isinstance(param, Ramp2DParam):
        param.a = 1.0
        param.b = 2.0
        param.c = 3.0
        param.xmin = -1.0
        param.xmax = 2.0
        param.ymin = -5.0
        param.ymax = 4.0
    elif isinstance(param, Gauss2DParam):
        param.x0 = param.y0 = 3.0
        param.sigma = 5.0
    elif isinstance(param, Checkerboard2DParam):
        param.square_size = 32
        param.vmin = 0.0
        param.vmax = 100.0
    elif isinstance(param, SinusoidalGrating2DParam):
        param.fx = 0.05
        param.fy = 0.0
        param.a = 50.0
        param.c = 100.0
    elif isinstance(param, Ring2DParam):
        param.period = 30.0
        param.ring_width = 10.0
    elif isinstance(param, SiemensStar2DParam):
        param.n_spokes = 24
        param.inner_radius = 10.0
        param.outer_radius = 80.0
    elif isinstance(param, Sinc2DParam):
        param.sigma = 15.0
        param.a = 100.0


def postprocess_image_object(
    obj: sigima.objects.ImageObj, itype: sigima.objects.ImageTypes
) -> None:
    """Postprocess the image object after creation.

    Args:
        obj: The image object to postprocess.
        itype: The type of the image.
    """
    if itype == sigima.objects.ImageTypes.ZEROS:
        assert (obj.data == 0).all()
    elif itype == sigima.objects.ImageTypes.RAMP:
        assert obj.data is not None
        check_scalar_result("Top-left corner", obj.data[0][0], -8.0)
        check_scalar_result("Top-right corner", obj.data[0][-1], -5.0)
        check_scalar_result("Bottom-left corner", obj.data[-1][0], 10.0)
        check_scalar_result("Bottom-right", obj.data[-1][-1], 13.0)
    else:
        assert obj.data is not None


def test_all_image_types() -> None:
    """Testing image creation from parameters"""
    execenv.print(f"{test_all_image_types.__doc__}:")
    for image in iterate_image_creation(
        preproc=preprocess_image_parameters,
        postproc=postprocess_image_object,
    ):
        assert image.data is not None
    execenv.print(f"{test_all_image_types.__doc__}: OK")


def __get_filenames_and_images() -> list[tuple[str, sigima.objects.ImageObj]]:
    """Get test filenames and images from the registry"""
    fi_list = [
        (fname, obj)
        for fname, obj in read_test_objects(ImageIORegistry)
        if obj is not None
    ]
    fi_list.append(("test_image_with_metadata", create_test_image_with_metadata()))
    fi_list.append(("annotated_image", create_annotated_image()))
    return fi_list


def test_hdf5_image_io() -> None:
    """Test HDF5 I/O for image objects with uniform and non-uniform coordinates"""
    execenv.print(f"{test_hdf5_image_io.__doc__}:")
    with WorkdirRestoringTempDir() as tmpdir:
        for fname, orig_image in __get_filenames_and_images():
            if orig_image is None:
                execenv.print(f"  Skipping {fname} (not implemented)")
                continue

            # Test Case 1: Original image with uniform coordinates (default)
            filename = osp.join(tmpdir, f"test_{osp.basename(fname)}_uniform.h5ima")
            sigima.io.write_image(filename, orig_image)
            execenv.print(f"  Saved {filename} (uniform coords)")

            # Read back
            fetch_image = sigima.io.read_image(filename)
            execenv.print(f"  Read {filename}")

            # Verify data
            data = fetch_image.data
            orig_data = orig_image.data
            assert isinstance(data, np.ndarray)
            assert isinstance(orig_data, np.ndarray)
            assert data.shape == orig_data.shape
            assert data.dtype == orig_data.dtype
            assert fetch_image.annotations == orig_image.annotations
            assert np.allclose(data, orig_data, atol=0.0, equal_nan=True)
            compare_metadata(
                fetch_image.metadata, orig_image.metadata.copy(), raise_on_diff=True
            )

            # Verify uniform coordinate attributes are preserved
            if orig_image.is_uniform_coords:
                assert fetch_image.is_uniform_coords
                assert fetch_image.dx == orig_image.dx
                assert fetch_image.dy == orig_image.dy
                assert fetch_image.x0 == orig_image.x0
                assert fetch_image.y0 == orig_image.y0
                execenv.print("    ✓ Uniform coordinates preserved")

            # Test Case 2: Same image with non-uniform coordinates
            # Create a modified version with non-uniform coordinates
            nonuniform_image = sigima.objects.create_image(
                title=orig_image.title + " (non-uniform)",
                data=orig_image.data.copy(),
                metadata=orig_image.metadata.copy(),
                units=(orig_image.xunit, orig_image.yunit, orig_image.zunit),
                labels=(orig_image.xlabel, orig_image.ylabel, orig_image.zlabel),
            )
            # Set non-uniform coordinates
            ny, nx = nonuniform_image.data.shape
            xcoords = np.linspace(0, 1, nx)
            ycoords = np.linspace(0, 1, ny) ** 2  # Quadratic spacing
            nonuniform_image.set_coords(xcoords=xcoords, ycoords=ycoords)

            # Save non-uniform version
            filename_nu = osp.join(
                tmpdir, f"test_{osp.basename(fname)}_nonuniform.h5ima"
            )
            sigima.io.write_image(filename_nu, nonuniform_image)
            execenv.print(f"  Saved {filename_nu} (non-uniform coords)")

            # Read back
            fetch_image_nu = sigima.io.read_image(filename_nu)
            execenv.print(f"  Read {filename_nu}")

            # Verify data
            assert np.allclose(
                fetch_image_nu.data, nonuniform_image.data, atol=0.0, equal_nan=True
            )

            # Verify non-uniform coordinate attributes are preserved
            assert not fetch_image_nu.is_uniform_coords
            assert np.array_equal(fetch_image_nu.xcoords, xcoords)
            assert np.array_equal(fetch_image_nu.ycoords, ycoords)
            execenv.print("    ✓ Non-uniform coordinates preserved")

    execenv.print(f"{test_hdf5_image_io.__doc__}: OK")


@pytest.mark.gui
def test_image_parameters_interactive() -> None:
    """Test interactive creation of image parameters"""
    execenv.print(f"{test_image_parameters_interactive.__doc__}:")
    with guiutils.lazy_qt_app_context(force=True):
        for itype in sigima.objects.ImageTypes:
            param = sigima.objects.create_image_parameters(itype)
            if param.edit():
                execenv.print(f"  Edited parameters for {itype.value}:")
                execenv.print(f"    {param}")
            else:
                execenv.print(f"  Skipped editing parameters for {itype.value}")
    execenv.print(f"{test_image_parameters_interactive.__doc__}: OK")


def test_create_image() -> None:
    """Test creation of an image object using `create_image` function"""
    execenv.print(f"{test_create_image.__doc__}:")
    # pylint: disable=import-outside-toplevel

    # Test all combinations of input parameters
    title = "Some Image"
    data = np.random.rand(10, 10)
    metadata = {"key": "value"}
    units = ("x unit", "y unit", "z unit")
    labels = ("x label", "y label", "z label")

    # 1. Create image with all parameters, and uniform coordinates
    image = sigima.objects.create_image(
        title=title,
        data=data,
        metadata=metadata,
        units=units,
        labels=labels,
    )
    assert isinstance(image, sigima.objects.ImageObj)
    assert image.title == title
    assert image.data is data  # Data should be the same object (not a copy)
    assert image.metadata == metadata
    assert (image.xunit, image.yunit, image.zunit) == units
    assert (image.xlabel, image.ylabel, image.zlabel) == labels
    dx, dy, x0, y0 = 0.1, 0.2, 50.0, 100.0
    image.set_uniform_coords(dx, dy, x0=x0, y0=y0)
    assert image.is_uniform_coords
    assert image.dx == dx
    assert image.dy == dy
    assert image.x0 == x0
    assert image.y0 == y0

    guiutils.view_images_if_gui(image, title=title)

    # 2. Create image with non-uniform coordinates
    xcoords = np.linspace(0, 1, 10)
    ycoords = np.linspace(0, 1, 10) ** 2
    image.set_coords(xcoords=xcoords, ycoords=ycoords)
    assert not image.is_uniform_coords
    assert np.array_equal(image.xcoords, xcoords)
    assert np.array_equal(image.ycoords, ycoords)

    guiutils.view_images_if_gui(image, title=title + " (non-uniform coords)")

    # 3. Create image with only data
    image = sigima.objects.create_image("", data=data)
    assert isinstance(image, sigima.objects.ImageObj)
    assert np.array_equal(image.data, data)
    assert not image.metadata
    assert (image.xunit, image.yunit, image.zunit) == ("", "", "")
    assert (image.xlabel, image.ylabel, image.zlabel) == ("", "", "")

    execenv.print(f"{test_create_image.__doc__}: OK")


def test_create_image_from_param() -> None:
    """Test creation of an image object using `create_image_from_param` function"""
    execenv.print(f"{test_create_image_from_param.__doc__}:")

    # Test 1: Basic parameter with defaults
    param = sigima.objects.NewImageParam()
    param.title = "Test Image"
    param.height = 100
    param.width = 200
    param.dtype = sigima.objects.ImageDatatypes.UINT16

    image = sigima.objects.create_image_from_param(param)
    assert isinstance(image, sigima.objects.ImageObj)
    assert image.title == "Test Image"
    assert image.data is not None
    assert image.data.shape == (100, 200)
    assert image.data.dtype == np.uint16
    assert (image.data == 0).all()  # NewImageParam generates zeros by default

    # Test 2: Parameter with default values (no explicit setting)
    param_defaults = sigima.objects.NewImageParam()
    # Don't set any values, use defaults

    image_defaults = sigima.objects.create_image_from_param(param_defaults)
    assert isinstance(image_defaults, sigima.objects.ImageObj)
    assert image_defaults.data is not None
    assert image_defaults.data.shape == (1024, 1024)  # Default dimensions
    assert image_defaults.data.dtype == np.float64  # Default dtype from NewImageParam

    # Test 3: Different image types using create_image_parameters
    test_cases = [
        (sigima.objects.ImageTypes.ZEROS, sigima.objects.ImageDatatypes.UINT8),
        (
            sigima.objects.ImageTypes.UNIFORM_DISTRIBUTION,
            sigima.objects.ImageDatatypes.FLOAT32,
        ),
        (
            sigima.objects.ImageTypes.NORMAL_DISTRIBUTION,
            sigima.objects.ImageDatatypes.FLOAT64,
        ),
        (sigima.objects.ImageTypes.GAUSS, sigima.objects.ImageDatatypes.UINT16),
        (sigima.objects.ImageTypes.RAMP, sigima.objects.ImageDatatypes.FLOAT64),
    ]

    for img_type, dtype in test_cases:
        param_type = sigima.objects.create_image_parameters(
            img_type,
            title=f"Test {img_type.value}",
            height=50,
            width=60,
            idtype=dtype,
        )

        # Preprocess parameters for specific types
        preprocess_image_parameters(param_type)

        image_type = sigima.objects.create_image_from_param(param_type)
        assert isinstance(image_type, sigima.objects.ImageObj)
        assert image_type.data is not None
        assert image_type.data.shape == (50, 60)
        assert image_type.data.dtype == dtype.value

        # Validate image type-specific properties
        if img_type == sigima.objects.ImageTypes.ZEROS:
            assert (image_type.data == 0).all()
        elif img_type == sigima.objects.ImageTypes.UNIFORM_DISTRIBUTION:
            # Uniform distribution should have varying values
            assert not (image_type.data == image_type.data[0, 0]).all()
            assert np.isfinite(image_type.data).all()
        elif img_type == sigima.objects.ImageTypes.NORMAL_DISTRIBUTION:
            # Normal distribution should have reasonable values
            assert not (image_type.data == 0).all()
            assert np.isfinite(image_type.data).all()
        elif img_type == sigima.objects.ImageTypes.GAUSS:
            # 2D Gaussian should have non-zero values
            assert not (image_type.data == 0).all()
            assert np.isfinite(image_type.data).all()
        elif img_type == sigima.objects.ImageTypes.RAMP:
            # Ramp should have varying values
            assert not (image_type.data == image_type.data[0, 0]).all()
            assert np.isfinite(image_type.data).all()

        # Test automatic title generation for distribution types
        if "DISTRIBUTION" in img_type.name:
            param_autotitle = sigima.objects.create_image_parameters(
                img_type, title="", height=50, width=60, idtype=dtype
            )
            image_autotitle = sigima.objects.create_image_from_param(param_autotitle)
            assert "Random" in image_autotitle.title, (
                f"Auto-generated title should contain 'Random' for {img_type.value}"
            )

    # Test 4: Gaussian parameters with specific values
    gauss_param = sigima.objects.Gauss2DParam()
    gauss_param.title = "Custom Gauss"
    gauss_param.height = 80
    gauss_param.width = 80
    gauss_param.dtype = sigima.objects.ImageDatatypes.FLOAT32

    gauss_image = sigima.objects.create_image_from_param(gauss_param)
    assert isinstance(gauss_image, sigima.objects.ImageObj)
    assert gauss_image.title == "Custom Gauss"
    assert gauss_image.data.shape == (80, 80)
    assert gauss_image.data.dtype == np.float32
    # Center should have highest value for Gaussian
    center_val = gauss_image.data[40, 40]
    corner_val = gauss_image.data[0, 0]
    assert center_val > corner_val

    # Test 5: Ramp parameters with specific values
    ramp_param = sigima.objects.Ramp2DParam()
    ramp_param.title = "Custom Ramp"
    ramp_param.height = 60
    ramp_param.width = 40
    ramp_param.dtype = sigima.objects.ImageDatatypes.FLOAT64

    ramp_image = sigima.objects.create_image_from_param(ramp_param)
    assert isinstance(ramp_image, sigima.objects.ImageObj)
    assert ramp_image.title == "Custom Ramp"
    assert ramp_image.data.shape == (60, 40)
    assert ramp_image.data.dtype == np.float64
    # Ramp should have different values at different positions
    assert ramp_image.data[0, 0] != ramp_image.data[-1, -1]

    execenv.print(f"{test_create_image_from_param.__doc__}: OK")


def test_image_copy() -> None:
    """Test copying image objects with uniform and non-uniform coordinates"""
    execenv.print(f"{test_image_copy.__doc__}:")

    # Create a base image with some data
    data = np.random.rand(50, 60)
    title = "Original Image"
    metadata = {"key1": "value1", "key2": 42}
    units = ("mm", "mm", "intensity")
    labels = ("X axis", "Y axis", "Intensity")

    # Test 1: Copy image with uniform coordinates
    execenv.print("  Test 1: Copy image with uniform coordinates")
    image_uniform = sigima.objects.create_image(
        title=title,
        data=data.copy(),
        metadata=metadata.copy(),
        units=units,
        labels=labels,
    )
    dx, dy, x0, y0 = 0.5, 0.8, 10.0, 20.0
    image_uniform.set_uniform_coords(dx, dy, x0=x0, y0=y0)
    # Set some scale attributes
    image_uniform.autoscale = False
    image_uniform.xscalelog = True
    image_uniform.xscalemin = 5.0
    image_uniform.xscalemax = 25.0
    image_uniform.yscalelog = False
    image_uniform.yscalemin = 15.0
    image_uniform.yscalemax = 35.0
    image_uniform.zscalemin = 0.0
    image_uniform.zscalemax = 1.0

    # Copy the image
    copied_uniform = image_uniform.copy()

    # Verify the copy
    assert copied_uniform is not image_uniform
    assert copied_uniform.title == image_uniform.title
    assert np.array_equal(copied_uniform.data, image_uniform.data)
    assert copied_uniform.data is not image_uniform.data  # Different array objects
    assert copied_uniform.metadata == image_uniform.metadata
    assert copied_uniform.metadata is not image_uniform.metadata
    assert (copied_uniform.xunit, copied_uniform.yunit, copied_uniform.zunit) == units
    assert (
        copied_uniform.xlabel,
        copied_uniform.ylabel,
        copied_uniform.zlabel,
    ) == labels

    # Verify uniform coordinates are preserved
    assert copied_uniform.is_uniform_coords == image_uniform.is_uniform_coords
    assert copied_uniform.is_uniform_coords is True
    assert copied_uniform.dx == dx
    assert copied_uniform.dy == dy
    assert copied_uniform.x0 == x0
    assert copied_uniform.y0 == y0
    execenv.print("    ✓ Uniform coordinates correctly copied")

    # Verify scale attributes are preserved
    assert copied_uniform.autoscale == image_uniform.autoscale
    assert copied_uniform.xscalelog == image_uniform.xscalelog
    assert copied_uniform.xscalemin == image_uniform.xscalemin
    assert copied_uniform.xscalemax == image_uniform.xscalemax
    assert copied_uniform.yscalelog == image_uniform.yscalelog
    assert copied_uniform.yscalemin == image_uniform.yscalemin
    assert copied_uniform.yscalemax == image_uniform.yscalemax
    assert copied_uniform.zscalemin == image_uniform.zscalemin
    assert copied_uniform.zscalemax == image_uniform.zscalemax
    execenv.print("    ✓ Scale attributes correctly copied")

    # Test 2: Copy image with non-uniform coordinates
    execenv.print("  Test 2: Copy image with non-uniform coordinates")
    image_nonuniform = sigima.objects.create_image(
        title=title + " (non-uniform)",
        data=data.copy(),
        metadata=metadata.copy(),
        units=units,
        labels=labels,
    )
    # Create non-uniform coordinates (quadratic spacing)
    ny, nx = data.shape
    xcoords = np.linspace(0, 10, nx) ** 1.5
    ycoords = np.linspace(0, 20, ny) ** 2
    image_nonuniform.set_coords(xcoords=xcoords, ycoords=ycoords)

    # Copy the image
    copied_nonuniform = image_nonuniform.copy()

    # Verify the copy
    assert copied_nonuniform is not image_nonuniform
    assert copied_nonuniform.title == image_nonuniform.title
    assert np.array_equal(copied_nonuniform.data, image_nonuniform.data)
    assert copied_nonuniform.data is not image_nonuniform.data
    assert copied_nonuniform.metadata == image_nonuniform.metadata
    assert copied_nonuniform.metadata is not image_nonuniform.metadata

    # Verify non-uniform coordinates are preserved
    assert copied_nonuniform.is_uniform_coords == image_nonuniform.is_uniform_coords
    assert copied_nonuniform.is_uniform_coords is False
    assert np.array_equal(copied_nonuniform.xcoords, xcoords)
    assert np.array_equal(copied_nonuniform.ycoords, ycoords)
    assert copied_nonuniform.xcoords is not image_nonuniform.xcoords
    assert copied_nonuniform.ycoords is not image_nonuniform.ycoords
    execenv.print("    ✓ Non-uniform coordinates correctly copied")

    # Test 3: Copy with title override
    execenv.print("  Test 3: Copy with custom title")
    new_title = "Copied Image"
    copied_with_title = image_uniform.copy(title=new_title)
    assert copied_with_title.title == new_title
    assert copied_with_title.is_uniform_coords is True
    assert copied_with_title.dx == dx
    execenv.print("    ✓ Title override works correctly")

    # Test 4: Copy with dtype conversion
    execenv.print("  Test 4: Copy with dtype conversion")
    copied_uint16 = image_uniform.copy(dtype=np.uint16)
    assert copied_uint16.data.dtype == np.uint16
    assert copied_uint16.is_uniform_coords is True
    assert copied_uint16.dx == dx
    execenv.print("    ✓ Dtype conversion works correctly")

    # Test 5: Copy with metadata filtering
    execenv.print("  Test 5: Copy with metadata filtering")
    copied_basic_meta = image_uniform.copy(all_metadata=False)
    assert copied_basic_meta.is_uniform_coords is True
    assert copied_basic_meta.dx == dx
    execenv.print("    ✓ Metadata filtering works correctly")

    execenv.print(f"{test_image_copy.__doc__}: OK")


def test_coordinate_conversion() -> None:
    """Test physical_to_indices and indices_to_physical methods"""
    execenv.print(f"{test_coordinate_conversion.__doc__}:")

    # Create a test image
    data = np.random.rand(100, 150)

    # ==================== Test 1: Uniform coordinates ====================
    execenv.print("  Test 1: Uniform coordinates - basic conversion")
    image_uniform = sigima.objects.create_image(
        title="Uniform Coordinates Test", data=data.copy()
    )
    dx, dy, x0, y0 = 0.5, 0.8, 10.0, 20.0
    image_uniform.set_uniform_coords(dx, dy, x0=x0, y0=y0)

    # Test basic forward conversion (physical → indices)
    physical_coords = [10.0, 20.0, 15.0, 30.0]  # Two points
    indices = image_uniform.physical_to_indices(physical_coords)
    assert len(indices) == 4
    assert indices[0] == 0  # (10.0 - 10.0) / 0.5 = 0
    assert indices[1] == 0  # (20.0 - 20.0) / 0.8 = 0
    assert indices[2] == 10  # (15.0 - 10.0) / 0.5 = 10
    assert indices[3] == 13  # (30.0 - 20.0) / 0.8 = 12.5 → 13 (floor(12.5 + 0.5))
    execenv.print("    ✓ Forward conversion (physical → indices) correct")

    # Test basic backward conversion (indices → physical)
    indices_input = [0, 0, 10, 12]
    coords = image_uniform.indices_to_physical(indices_input)
    assert len(coords) == 4
    assert coords[0] == 10.0  # 0 * 0.5 + 10.0 = 10.0
    assert coords[1] == 20.0  # 0 * 0.8 + 20.0 = 20.0
    assert coords[2] == 15.0  # 10 * 0.5 + 10.0 = 15.0
    assert coords[3] == 29.6  # 12 * 0.8 + 20.0 = 29.6
    execenv.print("    ✓ Backward conversion (indices → physical) correct")

    # Test round-trip accuracy
    execenv.print("  Test 2: Uniform coordinates - round-trip accuracy")
    original_physical = [12.5, 25.6, 18.3, 35.2]
    indices_rt = image_uniform.physical_to_indices(
        original_physical, as_float=True
    )  # Use float to preserve precision
    recovered_physical = image_uniform.indices_to_physical(indices_rt)
    np.testing.assert_allclose(recovered_physical, original_physical, rtol=1e-10)
    execenv.print("    ✓ Round-trip (physical → indices → physical) preserves values")

    # Test with origin offset and different pixel spacing
    execenv.print("  Test 3: Uniform coordinates - with non-zero origin")
    image_offset = sigima.objects.create_image(
        title="Offset Origin Test", data=data.copy()
    )
    image_offset.set_uniform_coords(dx=2.0, dy=3.0, x0=-5.0, y0=-10.0)
    phys = [-5.0, -10.0, 5.0, 20.0]
    idx = image_offset.physical_to_indices(phys)
    assert idx[0] == 0  # (-5.0 - (-5.0)) / 2.0 = 0
    assert idx[1] == 0  # (-10.0 - (-10.0)) / 3.0 = 0
    assert idx[2] == 5  # (5.0 - (-5.0)) / 2.0 = 5
    assert idx[3] == 10  # (20.0 - (-10.0)) / 3.0 = 10
    execenv.print("    ✓ Non-zero origin handled correctly")

    # Test clipping to image boundaries
    execenv.print("  Test 4: Uniform coordinates - clipping to boundaries")
    out_of_bounds = [-100.0, -100.0, 1000.0, 1000.0]
    clipped = image_uniform.physical_to_indices(out_of_bounds, clip=True)
    assert clipped[0] == 0  # Clipped to minimum X index
    assert clipped[1] == 0  # Clipped to minimum Y index
    assert clipped[2] == data.shape[1] - 1  # Clipped to maximum X index (149)
    assert clipped[3] == data.shape[0] - 1  # Clipped to maximum Y index (99)
    execenv.print("    ✓ Clipping to image boundaries works correctly")

    # Test as_float option
    execenv.print("  Test 5: Uniform coordinates - float indices")
    float_coords = [10.25, 20.4]
    float_indices = image_uniform.physical_to_indices(float_coords, as_float=True)
    int_indices = image_uniform.physical_to_indices(float_coords, as_float=False)
    assert isinstance(float_indices[0], float)
    assert isinstance(int_indices[0], (int, np.integer))
    assert float_indices[0] == 0.5  # (10.25 - 10.0) / 0.5 = 0.5
    assert int_indices[0] == 1  # floor(0.5 + 0.5) = 1
    execenv.print("    ✓ as_float option works correctly")

    # ==================== Test 6: Uniform to non-uniform conversion ==========
    execenv.print("  Test 6: Converting uniform to non-uniform coordinates")
    # Create a uniform image and test conversions
    image_to_convert = sigima.objects.create_image(
        title="Uniform to Non-uniform Test", data=data.copy()
    )
    dx_conv, dy_conv, x0_conv, y0_conv = 0.5, 0.8, 10.0, 20.0
    image_to_convert.set_uniform_coords(dx_conv, dy_conv, x0=x0_conv, y0=y0_conv)

    # Test conversions with uniform coordinates
    test_phys = [12.5, 25.6, 18.3, 35.2]
    indices_before = image_to_convert.physical_to_indices(test_phys, as_float=True)
    physical_before = image_to_convert.indices_to_physical([10.0, 20.0, 50.0, 60.0])

    # Convert to non-uniform coordinates
    image_to_convert.switch_coords_to("non-uniform")
    assert not image_to_convert.is_uniform_coords
    assert len(image_to_convert.xcoords) == data.shape[1]
    assert len(image_to_convert.ycoords) == data.shape[0]

    # Verify the generated xcoords and ycoords match the uniform grid
    expected_xcoords = np.linspace(
        x0_conv, x0_conv + dx_conv * (data.shape[1] - 1), data.shape[1]
    )
    expected_ycoords = np.linspace(
        y0_conv, y0_conv + dy_conv * (data.shape[0] - 1), data.shape[0]
    )
    np.testing.assert_allclose(image_to_convert.xcoords, expected_xcoords, rtol=1e-10)
    np.testing.assert_allclose(image_to_convert.ycoords, expected_ycoords, rtol=1e-10)
    execenv.print("    ✓ Generated non-uniform coords match uniform grid")

    # Test that conversions give the same results after switching to non-uniform
    indices_after = image_to_convert.physical_to_indices(test_phys, as_float=True)
    physical_after = image_to_convert.indices_to_physical([10.0, 20.0, 50.0, 60.0])
    np.testing.assert_allclose(indices_after, indices_before, rtol=1e-10)
    np.testing.assert_allclose(physical_after, physical_before, rtol=1e-10)
    execenv.print("    ✓ Coordinate conversions consistent after switch to non-uniform")

    # ==================== Test 7: Non-uniform coordinates ====================
    execenv.print("  Test 7: Non-uniform coordinates - basic conversion")
    image_nonuniform = sigima.objects.create_image(
        title="Non-Uniform Coordinates Test", data=data.copy()
    )

    # Create non-uniform coordinates with logarithmic spacing
    ny, nx = data.shape
    xcoords = np.logspace(0, 2, nx)  # 1 to 100, logarithmic spacing
    ycoords = np.linspace(0, 50, ny) ** 2  # 0 to 2500, quadratic spacing
    image_nonuniform.set_coords(xcoords=xcoords, ycoords=ycoords)

    # Test forward conversion with interpolation
    phys_nu = [xcoords[0], ycoords[0], xcoords[10], ycoords[20]]
    idx_nu = image_nonuniform.physical_to_indices(phys_nu, as_float=True)
    assert abs(idx_nu[0] - 0.0) < 1e-10  # First X coord → index 0
    assert abs(idx_nu[1] - 0.0) < 1e-10  # First Y coord → index 0
    assert abs(idx_nu[2] - 10.0) < 1e-10  # 10th X coord → index 10
    assert abs(idx_nu[3] - 20.0) < 1e-10  # 20th Y coord → index 20
    execenv.print("    ✓ Non-uniform forward conversion correct")

    # Test backward conversion with interpolation
    idx_back = [0.0, 0.0, 10.0, 20.0]
    coords_back = image_nonuniform.indices_to_physical(idx_back)
    assert abs(coords_back[0] - xcoords[0]) < 1e-10
    assert abs(coords_back[1] - ycoords[0]) < 1e-10
    assert abs(coords_back[2] - xcoords[10]) < 1e-10
    assert abs(coords_back[3] - ycoords[20]) < 1e-10
    execenv.print("    ✓ Non-uniform backward conversion correct")

    # Test round-trip for non-uniform coordinates
    execenv.print("  Test 8: Non-uniform coordinates - round-trip accuracy")
    original_nu = [xcoords[5], ycoords[15], xcoords[50], ycoords[75]]
    indices_nu_rt = image_nonuniform.physical_to_indices(original_nu, as_float=True)
    recovered_nu = image_nonuniform.indices_to_physical(indices_nu_rt)
    np.testing.assert_allclose(recovered_nu, original_nu, rtol=1e-10)
    execenv.print("    ✓ Round-trip for non-uniform coordinates preserves values")

    # Test interpolation between grid points
    execenv.print("  Test 9: Non-uniform coordinates - interpolation")
    # Test a coordinate between grid points
    mid_x = (xcoords[5] + xcoords[6]) / 2
    mid_y = (ycoords[10] + ycoords[11]) / 2
    mid_coords = [mid_x, mid_y]
    mid_indices = image_nonuniform.physical_to_indices(mid_coords, as_float=True)
    # Should be close to 5.5 and 10.5
    assert 5.4 < mid_indices[0] < 5.6
    assert 10.4 < mid_indices[1] < 10.6
    execenv.print("    ✓ Interpolation between grid points works")

    # ==================== Test 10: Edge cases ====================
    execenv.print("  Test 10: Edge cases")

    # Empty coordinate list
    empty_coords = []
    empty_indices = image_uniform.physical_to_indices(empty_coords)
    assert len(empty_indices) == 0
    execenv.print("    ✓ Empty coordinate list handled")

    # Single point
    single_point = [12.0, 25.0]
    single_idx = image_uniform.physical_to_indices(single_point)
    assert len(single_idx) == 2
    execenv.print("    ✓ Single point conversion works")

    # Multiple points
    multi_points = [10.0, 20.0, 15.0, 30.0, 20.0, 40.0, 25.0, 50.0]
    multi_idx = image_uniform.physical_to_indices(multi_points)
    assert len(multi_idx) == 8
    execenv.print("    ✓ Multiple points conversion works")

    # Odd number of coordinates should raise ValueError
    execenv.print("  Test 11: Error handling")
    try:
        image_uniform.physical_to_indices([10.0, 20.0, 15.0])  # Odd number
        assert False, "Should have raised ValueError for odd number of coords"
    except ValueError as e:
        assert "even number" in str(e)
        execenv.print("    ✓ ValueError raised for odd number of coordinates")

    try:
        image_uniform.indices_to_physical([0, 0, 5])  # Odd number
        assert False, "Should have raised ValueError for odd number of indices"
    except ValueError as e:
        assert "even number" in str(e)
        execenv.print("    ✓ ValueError raised for odd number of indices")

    # Test clipping with non-uniform coordinates
    execenv.print("  Test 12: Non-uniform coordinates - clipping")
    out_of_bounds_nu = [-1000.0, -1000.0, 10000.0, 10000.0]
    clipped_nu = image_nonuniform.physical_to_indices(out_of_bounds_nu, clip=True)
    assert clipped_nu[0] == 0
    assert clipped_nu[1] == 0
    assert clipped_nu[2] == data.shape[1] - 1
    assert clipped_nu[3] == data.shape[0] - 1
    execenv.print("    ✓ Clipping works for non-uniform coordinates")

    execenv.print(f"{test_coordinate_conversion.__doc__}: OK")


def test_switch_coords_errors() -> None:
    """Test switch_coords_to error handling with insufficient coordinates"""
    execenv.print(f"{test_switch_coords_errors.__doc__}:")

    # A single-column image only has one X coordinate, so switching its
    # non-uniform coordinates back to a uniform grid is impossible.
    data = np.random.rand(4, 1)
    image = sigima.objects.create_image(title="Single column", data=data)
    image.set_coords(xcoords=np.array([2.0]), ycoords=np.array([0.0, 1.0, 2.0, 3.0]))
    assert not image.is_uniform_coords

    try:
        image.switch_coords_to("uniform")
        assert False, "Should have raised ValueError for insufficient coordinates"
    except ValueError as exc:
        assert "not enough non-uniform coordinates" in str(exc)
        execenv.print("    ✓ ValueError raised when switching to uniform fails")

    # Switching a uniform image to non-uniform and back must round-trip exactly.
    image2 = sigima.objects.create_image(
        title="Round-trip", data=np.random.rand(10, 12)
    )
    image2.set_uniform_coords(dx=0.5, dy=0.8, x0=10.0, y0=20.0)
    image2.switch_coords_to("non-uniform")
    assert not image2.is_uniform_coords
    image2.switch_coords_to("uniform")
    assert image2.is_uniform_coords
    np.testing.assert_allclose([image2.x0, image2.y0], [10.0, 20.0], rtol=1e-10)
    np.testing.assert_allclose([image2.dx, image2.dy], [0.5, 0.8], rtol=1e-10)
    execenv.print("    ✓ Uniform → non-uniform → uniform round-trip preserved")

    execenv.print(f"{test_switch_coords_errors.__doc__}: OK")


if __name__ == "__main__":
    guiutils.enable_gui()
    test_create_image()
    test_image_parameters_interactive()
    test_all_image_types()
    test_hdf5_image_io()
    test_create_image_from_param()
    test_image_copy()
    test_coordinate_conversion()
    test_switch_coords_errors()
