# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.
"""
Convolution and Deconvolution
=============================

This example focuses on image blurring and sharpening using convolution
and deconvolution techniques provided by Sigima. Using various kernels,
we'll explore how these operations affect images and how to implement them
using Sigima's processing functions.

The example shows:

* Creating test images and kernels
* Basic convolution with Gaussian kernel
* Identity convolution (preserving original image)
* Deconvolution operations
* Effects of different kernel parameters
* Custom edge detection and sharpening kernels

This tutorial uses PlotPy for visualization, providing interactive plots
that allow you to explore the convolution results in detail.
"""

# %%
# Importing necessary modules
# ---------------------------
# We'll start by importing all the required modules for image processing
# and visualization.

import numpy as np
import scipy.signal

import sigima.objects
import sigima.params
import sigima.proc.image
from sigima import viz

# %%
# Creating test images and kernels
# --------------------------------
# We start by creating a test image and various convolution kernels.

# Set the fixed image size for this tutorial
size = 128

# Generate a test square image with a rectangle in the center
data = np.zeros((size, size), dtype=np.float64)
data[size // 5 : 2 * size // 5, size // 7 : 5 * size // 7] = 1.0
original_image = sigima.objects.create_image("Original Rectangle", data)

# Generate a Gaussian kernel
gparam = sigima.objects.Gauss2DParam.create(height=31, width=31, sigma=2.0)
nparam = sigima.params.NormalizeParam.create(method="area")
gaussian_kernel = sigima.objects.create_image_from_param(gparam)
gaussian_kernel = sigima.proc.image.normalize(gaussian_kernel, nparam)
gaussian_kernel.title = "Gaussian Kernel (σ=2.0)"

# Generate an identity kernel (impulse response)
identity_size = 15
identity_kernel = sigima.objects.create_image_from_param(
    sigima.objects.Zero2DParam.create(height=identity_size, width=identity_size)
)
identity_kernel.data[identity_size // 2, identity_size // 2] = 1.0
identity_kernel.title = "Identity Kernel"

print("✓ Test images and kernels created successfully!")
print("This example demonstrates convolution and deconvolution with Sigima.")
print(f"Original image shape: {original_image.data.shape}")
print(f"Gaussian kernel shape: {gaussian_kernel.data.shape}")
print(f"Identity kernel shape: {identity_kernel.data.shape}")

# Display the original image and kernels
viz.view_images_side_by_side(
    [original_image, gaussian_kernel, identity_kernel],
    ["Original Image", "Gaussian Kernel (σ=2.0)", "Identity Kernel"],
    title="Test Images and Kernels",
)

# %%
# Basic convolution with Gaussian kernel
# --------------------------------------
# Now we'll perform convolution with the Gaussian kernel and compare
# the result with scipy's implementation to verify correctness.

# Perform convolution with Gaussian kernel
convolved_gauss = sigima.proc.image.convolution(original_image, gaussian_kernel)
convolved_gauss.title = "Convolved with Gaussian"

# Compare with scipy implementation
expected_result = scipy.signal.convolve(
    original_image.data, gaussian_kernel.data, mode="same", method="auto"
)
print("\n✓ Convolution completed!")
max_diff = np.max(np.abs(convolved_gauss.data - expected_result))
print(f"Max difference from scipy: {max_diff:.2e}")

# Visualize the convolution process
viz.view_images_side_by_side(
    [original_image, gaussian_kernel, convolved_gauss],
    ["Original Image", "Gaussian Kernel (σ=2.0)", "Convolved Result"],
    title="Gaussian Convolution Example",
)

# %%
# Identity convolution
# --------------------
# Identity convolution should preserve the original image exactly.
# This demonstrates that our convolution implementation is working correctly.

# Perform convolution with identity kernel
convolved_identity = sigima.proc.image.convolution(original_image, identity_kernel)
convolved_identity.title = "Convolved with Identity"

# This should be nearly identical to the original
difference = np.max(np.abs(convolved_identity.data - original_image.data))
print("\n✓ Identity convolution completed!")
print(f"Max difference from original: {difference:.2e}")

# Visualize the identity convolution
viz.view_images_side_by_side(
    [original_image, identity_kernel, convolved_identity],
    ["Original Image", "Identity Kernel", "Convolved with Identity"],
    "Identity Convolution Example",
)

# %%
# Deconvolution with identity kernel
# ----------------------------------
# Deconvolution is the inverse operation of convolution. We'll start
# with a simple case using the identity kernel.

# Start with the convolved image and deconvolve using identity kernel
deconvolved_identity = sigima.proc.image.deconvolution(
    convolved_identity, identity_kernel
)
deconvolved_identity.title = "Deconvolved (Identity)"

# Check how well we recovered the original
recovery_error = np.max(np.abs(deconvolved_identity.data - original_image.data))
print("\n✓ Identity deconvolution completed!")
print(f"Recovery error: {recovery_error:.2e}")

# Visualize the deconvolution process
viz.view_images_side_by_side(
    [original_image, convolved_identity, deconvolved_identity],
    ["Original", "Convolved", "Deconvolved"],
    title="Identity Deconvolution Example",
)

# %%
# Advanced deconvolution with Gaussian kernel
# -------------------------------------------
# Now we'll try deconvolution with a Gaussian kernel, which is more
# challenging and demonstrates the limitations of deconvolution.

# Create a Gaussian kernel with smaller sigma for better deconvolution
gparam.sigma = 1.5
deconv_gaussian = sigima.proc.image.normalize(
    sigima.objects.create_image_from_param(gparam), nparam
)
deconv_gaussian.title = "Gaussian Kernel (σ=1.5)"

# Convolve the original image with this kernel
large_convolved = sigima.proc.image.convolution(original_image, deconv_gaussian)
large_convolved.title = "Convolved Image"

# Attempt deconvolution to recover the original
large_deconvolved = sigima.proc.image.deconvolution(large_convolved, deconv_gaussian)
large_deconvolved.title = "Deconvolved Result"

print("\n✓ Gaussian deconvolution completed!")
orig_min, orig_max = np.min(original_image.data), np.max(original_image.data)
deconv_min, deconv_max = np.min(large_deconvolved.data), np.max(large_deconvolved.data)
print(f"Original image range: [{orig_min:.3f}, {orig_max:.3f}]")
print(f"Deconvolved image range: [{deconv_min:.3f}, {deconv_max:.3f}]")

# Visualize the full deconvolution process
viz.view_images_side_by_side(
    [original_image, deconv_gaussian, large_convolved, large_deconvolved],
    ["Original", "Gaussian Kernel", "Convolved", "Deconvolved"],
    title="Gaussian Deconvolution Example",
)

# %%
# Exploring different kernel parameters
# -------------------------------------
# Different sigma values in Gaussian kernels produce different blurring effects.
# Let's compare several sigma values side by side.

# Create kernels with different sigma parameters
gparam.sigma = 0.8
small_sigma = sigima.proc.image.normalize(
    sigima.objects.create_image_from_param(gparam), nparam
)
gparam.sigma = 2.0
medium_sigma = sigima.proc.image.normalize(
    sigima.objects.create_image_from_param(gparam), nparam
)
gparam.sigma = 4.0
large_sigma = sigima.proc.image.normalize(
    sigima.objects.create_image_from_param(gparam), nparam
)

# Convolve the original image with each kernel
conv_small = sigima.proc.image.convolution(original_image, small_sigma)
conv_medium = sigima.proc.image.convolution(original_image, medium_sigma)
conv_large = sigima.proc.image.convolution(original_image, large_sigma)

print("\n✓ Multiple kernel comparison completed!")

# Show the effect of different sigma values on kernels
viz.view_images_side_by_side(
    [small_sigma, medium_sigma, large_sigma],
    ["Kernel σ=0.8", "Kernel σ=2.0", "Kernel σ=4.0"],
    title="Gaussian Kernels with Different Sigma Values",
)

# Show the effect of different sigma values on convolution results
viz.view_images_side_by_side(
    [conv_small, conv_medium, conv_large],
    ["Convolved σ=0.8", "Convolved σ=2.0", "Convolved σ=4.0"],
    title="Convolution Results with Different Sigma Values",
)

# %%
# Custom convolution kernels
# --------------------------
# Besides Gaussian kernels, we can create custom kernels for specific
# image processing tasks like edge detection and sharpening.

# Edge detection kernel (Sobel-like)
edge_data = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
edge_kernel = sigima.objects.create_image("Edge Detection Kernel", edge_data)

# Sharpening kernel
sharpen_data = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
sharpen_kernel = sigima.objects.create_image("Sharpening Kernel", sharpen_data)

# Apply custom kernels to the original image
edge_result = sigima.proc.image.convolution(original_image, edge_kernel)
edge_result.title = "Edge Detection"

sharpen_result = sigima.proc.image.convolution(original_image, sharpen_kernel)
sharpen_result.title = "Sharpened"

print("\n✓ Custom kernel convolutions completed!")

# Visualize custom kernels
viz.view_images_side_by_side(
    [edge_kernel, sharpen_kernel],
    ["Edge Detection Kernel", "Sharpening Kernel"],
    title="Custom Convolution Kernels",
)

# Visualize custom kernel results
viz.view_images_side_by_side(
    [original_image, edge_result, sharpen_result],
    ["Original", "Edge Detection", "Sharpened"],
    title="Custom Kernel Convolution Results",
)

# %%
# Summary and conclusions
# -----------------------
# This tutorial demonstrated the key concepts of convolution and deconvolution
# in image processing using Sigima.

print("\n" + "=" * 60)
print("CONVOLUTION TUTORIAL SUMMARY")
print("=" * 60)
print("✓ Created test images and various kernels")
print("✓ Demonstrated basic Gaussian convolution")
print("✓ Showed identity kernel behavior")
print("✓ Performed deconvolution operations")
print("✓ Explored different kernel parameters")
print("✓ Applied custom edge detection and sharpening kernels")
print("\nKey Takeaways:")
print("• Larger sigma values create more blurring")
print("• Identity kernels preserve the original image")
print("• Deconvolution can recover original features (with limitations)")
print("• Custom kernels enable specialized image processing effects")

# Final comparison showing the complete pipeline
dialog10 = viz.view_images_side_by_side(
    [original_image, gaussian_kernel, convolved_gauss, large_deconvolved],
    ["Original", "Gaussian Kernel", "Convolved", "Deconvolved"],
    title="Complete Convolution-Deconvolution Pipeline",
)
