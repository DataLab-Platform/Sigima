"""
Uniform and Non-Uniform Coordinate Systems
==========================================

This example demonstrates the use of uniform and non-uniform coordinate systems
with images in Sigima. It shows how to create, visualize, and work with both
types of coordinate systems, highlighting their differences and appropriate
use cases.

The example shows:

* Creating images with uniform coordinate systems
* Creating images with non-uniform coordinate systems
* Visualizing the coordinate grids
* Comparing pixel spacing and coordinate mapping
* Working with real-world units and coordinates
* Understanding when to use each coordinate system type

This tutorial uses PlotPy for visualization, providing interactive plots
that allow you to explore the coordinate system effects in detail.
"""

# %%
# Importing necessary modules
# ---------------------------
# We'll start by importing all the required modules for image processing
# and visualization.

import numpy as np

from sigima.objects import create_image
from sigima.tests.vistools import view_images_side_by_side

# %%
# Creating test images with uniform coordinates
# ---------------------------------------------
# Uniform coordinates have constant spacing between pixels in both directions.
# This is the most common case for regular imaging systems.

# Create a simple test pattern - a sine wave pattern
size = 100
x_uniform = np.linspace(0, 4 * np.pi, size)
y_uniform = np.linspace(0, 4 * np.pi, size)
X_uniform, Y_uniform = np.meshgrid(x_uniform, y_uniform)

# Create a 2D sine wave pattern
data_uniform = np.sin(X_uniform) * np.cos(Y_uniform)

# Create the image object with uniform coordinates
uniform_image = create_image(
    title="Uniform Coordinates",
    data=data_uniform,
    units=("μm", "μm", "intensity"),
    labels=("X position", "Y position", "Signal"),
)

# Set uniform coordinate system with specific spacing and origin
dx = 0.1  # 0.1 μm per pixel in X
dy = 0.1  # 0.1 μm per pixel in Y
x0 = -2.0  # X origin at -2.0 μm
y0 = -2.0  # Y origin at -2.0 μm

uniform_image.set_uniform_coords(dx, dy, x0=x0, y0=y0)

print("✓ Uniform coordinate image created!")
print(f"Image shape: {uniform_image.data.shape}")
coord_type = "Uniform" if uniform_image.is_uniform_coords else "Non-uniform"
print(f"Coordinate system: {coord_type}")
print(f"X spacing (dx): {uniform_image.dx} μm")
print(f"Y spacing (dy): {uniform_image.dy} μm")
print(f"X origin: {uniform_image.x0} μm")
print(f"Y origin: {uniform_image.y0} μm")
x_end = uniform_image.x0 + uniform_image.dx * (uniform_image.data.shape[1] - 1)
y_end = uniform_image.y0 + uniform_image.dy * (uniform_image.data.shape[0] - 1)
print(f"X range: {uniform_image.x0:.1f} to {x_end:.1f} μm")
print(f"Y range: {uniform_image.y0:.1f} to {y_end:.1f} μm")

# %%
# Creating test images with non-uniform coordinates
# -------------------------------------------------
# Non-uniform coordinates allow for variable spacing between pixels,
# useful for adaptive sampling, curved coordinates, or irregular grids.

# Create the same sine wave pattern but with non-uniform coordinates
data_nonuniform = data_uniform.copy()

# Create non-uniform coordinate arrays
# X coordinates: denser near the center, sparser at edges
x_center = 2 * np.pi
x_range = 4 * np.pi
x_nonuniform = x_center + (x_range / 2) * np.tanh(np.linspace(-2, 2, size))

# Y coordinates: quadratic spacing (denser at bottom)
y_start = 0
y_end = 4 * np.pi
y_nonuniform = y_start + (y_end - y_start) * (np.linspace(0, 1, size) ** 2)

# Create the image object with non-uniform coordinates
nonuniform_image = create_image(
    title="Non-Uniform Coordinates",
    data=data_nonuniform,
    units=("μm", "μm", "intensity"),
    labels=("X position", "Y position", "Signal"),
)

# Set non-uniform coordinate system
nonuniform_image.set_coords(xcoords=x_nonuniform, ycoords=y_nonuniform)

print("\n✓ Non-uniform coordinate image created!")
print(f"Image shape: {nonuniform_image.data.shape}")
coord_type = "Uniform" if nonuniform_image.is_uniform_coords else "Non-uniform"
print(f"Coordinate system: {coord_type}")
print(f"X coordinates range: {x_nonuniform[0]:.3f} to {x_nonuniform[-1]:.3f} μm")
print(f"Y coordinates range: {y_nonuniform[0]:.3f} to {y_nonuniform[-1]:.3f} μm")
x_spacing_min = np.min(np.diff(x_nonuniform))
x_spacing_max = np.max(np.diff(x_nonuniform))
y_spacing_min = np.min(np.diff(y_nonuniform))
y_spacing_max = np.max(np.diff(y_nonuniform))
print(f"X spacing varies from {x_spacing_min:.4f} to {x_spacing_max:.4f} μm")
print(f"Y spacing varies from {y_spacing_min:.4f} to {y_spacing_max:.4f} μm")

# %%
# Visualizing coordinate system differences
# -----------------------------------------
# Let's create coordinate grid visualizations to highlight the differences
# between uniform and non-uniform coordinate systems.

# Create coordinate grid images for visualization
grid_uniform = np.zeros_like(data_uniform)
grid_nonuniform = np.zeros_like(data_nonuniform)

# Add grid lines every 10 pixels for uniform coordinates
grid_uniform[::10, :] = 1.0  # Horizontal lines
grid_uniform[:, ::10] = 1.0  # Vertical lines

# Add grid lines every 10 pixels for non-uniform coordinates
grid_nonuniform[::10, :] = 1.0  # Horizontal lines
grid_nonuniform[:, ::10] = 1.0  # Vertical lines

# Create grid visualization images
uniform_grid = create_image(
    title="Uniform Grid",
    data=grid_uniform,
    units=("μm", "μm", "grid"),
    labels=("X position", "Y position", "Grid lines"),
)
uniform_grid.set_uniform_coords(dx, dy, x0=x0, y0=y0)

nonuniform_grid = create_image(
    title="Non-Uniform Grid",
    data=grid_nonuniform,
    units=("μm", "μm", "grid"),
    labels=("X position", "Y position", "Grid lines"),
)
nonuniform_grid.set_coords(xcoords=x_nonuniform, ycoords=y_nonuniform)

print("\n✓ Coordinate grid visualizations created!")

# Display the coordinate system comparison
view_images_side_by_side(
    [uniform_image, nonuniform_image],
    ["Uniform Coordinates", "Non-Uniform Coordinates"],
    title="Coordinate Systems Comparison - Data Images",
    share_axes=False,
)

view_images_side_by_side(
    [uniform_grid, nonuniform_grid],
    ["Uniform Grid", "Non-Uniform Grid"],
    title="Coordinate Systems Comparison - Grid Visualization",
    share_axes=False,
)

# %%
# Creating specialized non-uniform coordinate examples
# ----------------------------------------------------
# Let's create some more realistic examples of non-uniform coordinates
# that might be encountered in real applications:
#
# 1. Time-resolved spectroscopy with logarithmic wavelength scale
# 2. Polar to Cartesian mapping

# Example 1: Logarithmic wavelength scale for spectroscopy
# Simulating a time-resolved spectroscopy measurement with log-spaced wavelengths
size_log = 80
wavelengths = np.logspace(np.log10(400), np.log10(800), size_log)  # 400-800 nm
time_points = np.linspace(0, 200, size_log)  # Time in milliseconds
W, T = np.meshgrid(wavelengths, time_points)

# Create a spectral response pattern: a peak that shifts over time
# Simulates fluorescence decay with spectral shift
peak_center = 500 + 0.5 * T  # Peak shifts from 500 to 600 nm over time
spectral_data = np.exp(-(((W - peak_center) / 40) ** 2)) * np.exp(-T / 100)

spectral_image = create_image(
    title="Time-Resolved Spectroscopy (Log λ)",
    data=spectral_data,
    units=("nm", "ms", "counts"),
    labels=("Wavelength", "Time", "Fluorescence"),
)
spectral_image.set_coords(xcoords=wavelengths, ycoords=time_points)

print("\n✓ Time-resolved spectroscopy example created!")
print(f"Wavelength range: {wavelengths[0]:.1f} to {wavelengths[-1]:.1f} nm")
print(f"Time range: {time_points[0]:.1f} to {time_points[-1]:.1f} ms")
wl_spacing_min = np.min(np.diff(wavelengths))
wl_spacing_max = np.max(np.diff(wavelengths))
print(f"Log wavelength spacing: {wl_spacing_min:.2f} to {wl_spacing_max:.2f} nm")

# Example 2: Polar to Cartesian mapping
size_polar = 60
r_coords = np.linspace(1, 10, size_polar)
theta_coords = np.linspace(0, 2 * np.pi, size_polar)

# Convert to Cartesian coordinates for non-uniform mapping
x_polar = np.zeros((size_polar, size_polar))
y_polar = np.zeros((size_polar, size_polar))

for i, r in enumerate(r_coords):
    for j, theta in enumerate(theta_coords):
        x_polar[i, j] = r * np.cos(theta)
        y_polar[i, j] = r * np.sin(theta)

# Create a radial pattern
polar_data = np.zeros((size_polar, size_polar))
for i, r in enumerate(r_coords):
    for j, theta in enumerate(theta_coords):
        polar_data[i, j] = np.sin(3 * theta) * np.exp(-r / 5)

# Note: For this example, we'll use the polar coordinates directly
# In practice, you might want to interpolate to a regular Cartesian grid
polar_image = create_image(
    title="Polar Coordinate Mapping",
    data=polar_data,
    units=("mm", "rad", "signal"),
    labels=("Radius", "Angle", "Amplitude"),
)
polar_image.set_coords(xcoords=r_coords, ycoords=theta_coords)

print("\n✓ Polar coordinate example created!")
print(f"Radial range: {r_coords[0]:.1f} to {r_coords[-1]:.1f} mm")
print(f"Angular range: {theta_coords[0]:.2f} to {theta_coords[-1]:.2f} rad")

# Display the specialized examples
view_images_side_by_side(
    [spectral_image, polar_image],
    ["Time-Resolved Spectroscopy (Log λ)", "Polar Coordinates"],
    title="Specialized Non-Uniform Coordinate Examples",
    share_axes=False,
)

# %%
# Summary and best practices
# --------------------------
# Let's summarize when to use each coordinate system type.

print("\n" + "=" * 60)
print("COORDINATE SYSTEMS SUMMARY")
print("=" * 60)

print("\n🔲 UNIFORM COORDINATES:")
print("   ✓ Regular imaging systems (cameras, microscopes)")
print("   ✓ Constant pixel spacing in physical units")
print("   ✓ Simple and memory-efficient")
print("   ✓ Fast computations and interpolations")
print("   ✓ Easy integration with standard image processing")

print("\n🔳 NON-UNIFORM COORDINATES:")
print("   ✓ Adaptive sampling systems")
print("   ✓ Curved or distorted coordinate systems")
print("   ✓ Logarithmic or specialized scales")
print("   ✓ Irregular measurement grids")
print("   ✓ Coordinate transformations (polar to Cartesian)")

print("\n📊 PERFORMANCE CONSIDERATIONS:")
print("   • Uniform: O(1) coordinate lookup")
print("   • Non-uniform: O(n) coordinate lookup")
print("   • Memory: Uniform uses 4 parameters, Non-uniform uses 2×N arrays")

print("\n💾 FILE FORMAT SUPPORT:")
print("   • Both coordinate types supported in Sigima HDF5 format")
print("   • Coordinated text format supports non-uniform coordinates")
print("   • Standard image formats (TIFF, etc.) assume uniform coordinates")

# Final comparison view
view_images_side_by_side(
    [uniform_image, nonuniform_image, spectral_image, polar_image],
    [
        "Uniform\n(Regular Grid)",
        "Non-Uniform\n(Variable Grid)",
        "Time-Resolved\n(Log Wavelength)",
        "Polar\n(Radial Data)",
    ],
    title="Complete Coordinate Systems Overview",
    share_axes=False,
)

print("\n✨ Example completed successfully!")
print("This demonstrates the flexibility of Sigima's coordinate system support.")
