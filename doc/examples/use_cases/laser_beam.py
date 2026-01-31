# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Laser Beam Size Measurement Example
===================================

This example demonstrates comprehensive laser beam analysis techniques following
the laser beam tutorial workflow. It shows how to load multiple laser beam images,
analyze background noise with histograms, apply proper clipping, detect beam
centroids, extract line and radial profiles, compute FWHM measurements, and
track beam size evolution along the propagation axis.

The script demonstrates advanced optical beam characterization workflows commonly
used in laser physics, beam quality assessment, and optical system design.
"""

# %%
# Importing necessary modules
# --------------------------------
# We start by importing all the required modules for image processing
# and visualization. To run this example, ensure you have all the required
# dependencies installed.

import numpy as np

import sigima.io
import sigima.objects
import sigima.params
import sigima.proc.image
import sigima.proc.signal
from sigima import viz
from sigima.tests import helpers

# %%
# Load all laser beam images
# ------------------------------------------------
# We load a series of laser beam images taken at different positions along
# the propagation axis (z-axis). The images are contained in the folder laser_beam and
# named following the pattern TEM00_z_*.jpg, where * is the z position in arbitrary
# units.


def load_laser_beam_images():
    """Load all laser beam test images from the test data directory.

    Returns:
        List of image objects loaded from TEM00_z_*.jpg files
    """
    # Get all TEM00 laser beam image files
    image_files = helpers.get_test_fnames("laser_beam/TEM00_z_*.jpg")

    # Sort files by z-position (extract number from filename)
    image_files.sort(key=lambda f: int(f.split("_z_")[1].split(".")[0]))

    # Load images
    images = []
    for filepath in image_files:
        img = sigima.io.read_image(filepath)
        # Extract z position from filename for proper naming
        z_pos = filepath.split("_z_")[1].split(".")[0]
        img.title = f"TEM00_z_{z_pos}"
        images.append(img)

    return images


images = load_laser_beam_images()

print(f"✓ Loaded {len(images)} laser beam images")
print("Image details:")
for i, img in enumerate(images):
    intensity_range = f"{img.data.min()}-{img.data.max()}"
    print(f"  {i + 1}. {img.title}: {img.data.shape}, range {intensity_range}")


# %%
# Visualize the first few images
print("\n✓ Visualizing sample images...")
viz.view_images_side_by_side(images[:3], title="Sample Laser Beam Images")

# %%
# Background noise analysis with histogram
# ------------------------------------------------
# To analyze the background noise characteristics of the laser beam images,
# we create a histogram of pixel values from the first image. This helps us
# identify the noise floor and determine an appropriate clipping threshold.


print("\n--- Background Noise Analysis ---")

hist_param = sigima.params.HistogramParam()
hist_param.bins = 100
hist_param.range = (0, images[0].data.max())

hist = sigima.proc.image.histogram(images[0], hist_param)
hist.title = "Pixel value histogram of image 1"

print(f"✓ Generated histogram with {hist_param.bins} bins")
print(f"Histogram range: {hist_param.range[0]} - {hist_param.range[1]}")
print("The histogram shows background noise distribution")

# Visualize histogram
viz.view_curves([hist], title="Pixel Value Histogram - Background Analysis")

# %%
# Based on the histogram analysis, we determine a clipping threshold around 30-35 LSB to
# effectively remove background noise from all images.

# %%
# Background noise removal via clipping
# ------------------------------------------------
# We set the threshold to 35 and we apply this clipping to each image in the dataset.

background_threshold = 35

print(f"Will use clipping threshold of {background_threshold} LSB")
# %%
# In order to perform the clipping, we create a ClipParam object
# and set the minimum value to the background threshold. We then apply
# the clipping to each image in the dataset.

print("\n--- Applying Background Clipping ---")
clip_param = sigima.params.ClipParam()
clip_param.lower = background_threshold  # Remove background noise below 35 LSB

clipped_images = []
for img in images:
    clipped_img = sigima.proc.image.clip(img, clip_param)
    clipped_img.title = f"{img.title}_clipped"
    clipped_images.append(clipped_img)

print(f"✓ Applied clipping of {clip_param.lower} LSB to all {len(images)} images")
print("Background noise below threshold has been removed")

# %%
# We can now visualize some clipped images:
viz.view_images_side_by_side(
    images[:3] + clipped_images[:3],
    rows=2,
    title="Original and Clipped Images (First 3)",
)

# %%
# Compute centroids for beam center detection
# ------------------------------------------------
# Next, we compute the centroid of each clipped image to determine the beam center. This
# is important for accurate profile extraction and FWHM measurements.

print("\n--- Computing Beam Centroids ---")

centroids = []
for img in clipped_images:
    centroid_result = sigima.proc.image.centroid(img)
    centroids.append(centroid_result.value)  # (x, y)
    print(f"  ✓ {img.title}: centroid at {centroid_result.value}")

print(f"\n✓ Successfully detected {len(centroids)}/{len(images)} centroids")

# %%
# Extract line profiles through beam centers
# ------------------------------------------------
# We extract horizontal line profiles through the detected centroids of each clipped
# image. This provides insight into the beam intensity distribution along a horizontal
# cross-section.

print("\n--- Extracting Line Profiles ---")

line_profiles = []
for i, (img, centroid_coords) in enumerate(zip(clipped_images, centroids)):
    # Create line profile parameters for horizontal line through centroid
    line_param = sigima.proc.image.LineProfileParam()
    line_param.direction = "horizontal"
    line_param.row = int(centroid_coords[1])  # Use centroid y-coordinate as row

    # Extract line profile
    profile = sigima.proc.image.line_profile(img, line_param)
    profile.title = f"Line_profile_{img.title}"
    line_profiles.append(profile)
    print(f"  ✓ Extracted line profile for {img.title} at row {line_param.row}")

print(f"\n✓ Generated {len(line_profiles)} line profiles")

# Visualize some line profiles
viz.view_curves(line_profiles[:3], title="Horizontal Line Profiles (First 3 Images)")

# %%
# Extract radial profiles around beam centers
# ------------------------------------------------
# We extract radial profiles centered on the detected centroids of each clipped image.
# This provides a circular intensity distribution useful for FWHM measurements.

print("\n--- Extracting Radial Profiles ---")

radial_profiles = []
for img in clipped_images:
    # Create radial profile parameters using automatic centroid detection
    radial_param = sigima.proc.image.RadialProfileParam()
    radial_param.center = "centroid"  # Use automatic centroid detection

    # Extract radial profile
    profile = sigima.proc.image.radial_profile(img, radial_param)
    profile.title = f"Radial_profile_{img.title}"
    radial_profiles.append(profile)
    print(f"  ✓ Extracted radial profile for {img.title}")

print(f"\n✓ Generated {len(radial_profiles)} radial profiles")

# %%
# We can now visualize some radial profiles
viz.view_curves(radial_profiles[:3], title="Radial Profiles (First 3 Images)")

# %%
# Compute FWHM for radial profiles
# ------------------------------------------------
# We compute the Full Width at Half Maximum (FWHM) for each radial profile to
# quantify the beam size. The FWHM is a standard metric for beam width.

print("\n--- Computing FWHM Measurements ---")

fwhm_vals = []
fwhm_param = sigima.params.FWHMParam()
fwhm_param.method = "zero-crossing"  # Standard FWHM method

for profile in radial_profiles:
    fwhm_result = sigima.proc.signal.fwhm(profile, fwhm_param)
    fwhm_vals.append(fwhm_result.value)
    print(f"  ✓ {profile.title}: FWHM = {fwhm_result.value:.2f} pixels")
# %%
# That's done, we can now print some FWHM statistics to check our results:
print("\n✓ FWHM Statistics:")
print(f"  Valid measurements: {len(fwhm_vals)}/{len(fwhm_vals)}")
print(f"  Beam size range: {min(fwhm_vals):.2f} - {max(fwhm_vals):.2f} pixels")
print(f"  Average beam size: {np.mean(fwhm_vals):.2f} ± {np.std(fwhm_vals):.2f} pixels")
# %%
# Everything seems fine, we can now analyze the beam size evolution along the z-axis.

# %%
# Analyze beam size evolution
# ------------------------------------------------
# Having computed the FWHM for each radial profile, we can now analyze how the
# beam size evolves along the propagation axis (z-axis). This is useful to extract some
# meaningful information from all the numbers we have obtained. We create a signal
# representing the beam size evolution and visualize it.

print("\n--- Beam Size Evolution Analysis ---")

# Create a signal showing beam size evolution along z-axis
z_positions = list(range(len(fwhm_vals)))
beam_evolution = sigima.objects.create_signal(
    "Beam size evolution",
    np.array(z_positions),
    np.array(fwhm_vals),
    units=("image_index", "pixels"),
)

print(f"✓ Created beam evolution signal with {len(z_positions)} data points")

# Visualize beam size evolution
viz.view_curves(
    [beam_evolution], title="Beam Size Evolution vs Z-Position (uncalibrated)"
)

# %%
# The beam evolution signal currently uses image index as the x-axis, and even if we can
# see the trend, it is not very informative. It would be way better to have the x-axis
# in mm.

# %%
# Apply z-axis calibration
# ------------------------------------------------
# We apply a linear calibration to the x-axis of the beam evolution signal to convert
# image index to physical distance in mm. In Sigima there is the possibility to perform
# a linear axis calibration. We use the formula x' = 5*x + 15, where x is
# the image index and x' is the calibrated distance in mm.

print("\n--- Applying Z-Axis Calibration ---")

# Calibrate x-axis using the formula: x' = 5*x + 15 (convert to mm)
calib_param = sigima.proc.signal.XYCalibrateParam()
calib_param.axis = "x"
calib_param.a = 5.0  # Scale factor
calib_param.b = 15.0  # Offset

beam_evolution_calibrated = sigima.proc.signal.calibration(beam_evolution, calib_param)
beam_evolution_calibrated.title = f"{beam_evolution.title} (z calibrated)"

print(f"✓ Applied calibration: z' = {calib_param.a}*z + {calib_param.b}")
print("Z-axis now represents physical distance in mm")

# Visualize calibrated beam evolution
viz.view_curves(
    [beam_evolution_calibrated],
    title="Beam Size Evolution vs Z-Position (calibrated)",
    xlabel="Z Position (mm)",
    ylabel="Beam Size (pixels)",
)
