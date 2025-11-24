"""Advanced post-processing effects for terrain renders."""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageFilter


__all__ = [
    "apply_tonemapping",
    "apply_color_grading",
    "apply_bloom_effect",
    "apply_ssao_approximation",
    "apply_sharpening",
    "apply_atmospheric_perspective",
]


def apply_tonemapping(
    img_array: np.ndarray,
    method: Literal["reinhard", "filmic", "aces", "uncharted2"] = "aces",
    exposure: float = 1.0,
) -> np.ndarray:
    """
    Apply tonemapping to HDR-like image data.

    Args:
        img_array: RGB image array (0-1 range or HDR values)
        method: Tonemapping algorithm to use
        exposure: Exposure adjustment factor

    Returns:
        Tonemapped RGB array (0-1 range)
    """
    img = img_array.astype(float) * exposure

    if method == "reinhard":
        # Simple Reinhard tonemapping
        result = img / (1.0 + img)

    elif method == "filmic":
        # Filmic curve (similar to Unreal Engine)
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        result = (img * (a * img + b)) / (img * (c * img + d) + e)

    elif method == "aces":
        # ACES approximation
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        result = np.clip((img * (a * img + b)) /
                         (img * (c * img + d) + e), 0, 1)

    elif method == "uncharted2":
        # Uncharted 2 tonemapping
        def uncharted2_tonemap_partial(x):
            A = 0.15
            B = 0.50
            C = 0.10
            D = 0.20
            E = 0.02
            F = 0.30
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

        exposure_bias = 2.0
        curr = uncharted2_tonemap_partial(img * exposure_bias)
        W = 11.2
        white_scale = 1.0 / uncharted2_tonemap_partial(np.array([W]))
        result = curr * white_scale

    else:
        result = img

    return np.clip(result, 0.0, 1.0)


def apply_color_grading(
    img_array: np.ndarray,
    temperature: float = 0.0,
    tint: float = 0.0,
    saturation: float = 1.0,
    contrast: float = 1.0,
    brightness: float = 0.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Apply color grading adjustments.

    Args:
        img_array: RGB image array (0-1 range)
        temperature: Color temperature shift (-1 to 1, warm/cool)
        tint: Tint adjustment (-1 to 1, green/magenta)
        saturation: Saturation multiplier
        contrast: Contrast multiplier
        brightness: Brightness offset
        gamma: Gamma correction

    Returns:
        Color graded RGB array
    """
    img = img_array.astype(float)

    # Temperature/tint adjustment
    if temperature != 0.0 or tint != 0.0:
        temp_matrix = np.array([
            [1.0 + temperature * 0.3, 0.0, 0.0],
            [0.0, 1.0 + tint * 0.3, 0.0],
            [0.0, 0.0, 1.0 - temperature * 0.3],
        ])

        if len(img.shape) == 3 and img.shape[2] >= 3:
            img_flat = img.reshape(-1, img.shape[2])
            img_flat = img_flat[:, :3] @ temp_matrix.T
            img[:, :, :3] = img_flat.reshape(img.shape[0], img.shape[1], 3)

    # Saturation
    if saturation != 1.0 and len(img.shape) == 3:
        gray = np.mean(img[:, :, :3], axis=2, keepdims=True)
        img = gray + (img - gray) * saturation

    # Brightness
    if brightness != 0.0:
        img = img + brightness

    # Contrast
    if contrast != 1.0:
        mid = 0.5
        img = (img - mid) * contrast + mid

    # Gamma
    if gamma != 1.0:
        img = np.power(np.clip(img, 0, 1), 1.0 / gamma)

    return np.clip(img, 0.0, 1.0)


def apply_bloom_effect(
    img_array: np.ndarray,
    threshold: float = 0.8,
    intensity: float = 0.3,
    blur_radius: float = 10.0,
) -> np.ndarray:
    """
    Apply bloom/glow effect to bright areas.

    Args:
        img_array: RGB image array (0-1 range)
        threshold: Brightness threshold for bloom
        intensity: Bloom intensity
        blur_radius: Blur radius for bloom

    Returns:
        Image with bloom effect applied
    """
    img = img_array.astype(float)

    # Extract bright areas
    if len(img.shape) == 3:
        brightness = np.mean(img, axis=2)
    else:
        brightness = img

    bright_mask = brightness > threshold
    bright_areas = img.copy()

    if len(img.shape) == 3:
        bright_areas[~bright_mask] = 0.0
    else:
        bright_areas = bright_areas * bright_mask

    # Blur bright areas
    if len(img.shape) == 3:
        for channel in range(img.shape[2]):
            bright_areas[:, :, channel] = ndimage.gaussian_filter(
                bright_areas[:, :, channel], sigma=blur_radius
            )
    else:
        bright_areas = ndimage.gaussian_filter(bright_areas, sigma=blur_radius)

    # Add bloom
    result = img + bright_areas * intensity

    return np.clip(result, 0.0, 1.0)


def apply_ssao_approximation(
    height_array: np.ndarray,
    radius: int = 5,
    intensity: float = 0.5,
) -> np.ndarray:
    """
    Apply screen-space ambient occlusion approximation based on heightmap.

    Args:
        height_array: Height values (0-1 range)
        radius: Sample radius for occlusion
        intensity: Occlusion intensity

    Returns:
        AO map (0-1 range, 0=fully occluded)
    """
    # Calculate local height variance as occlusion proxy
    local_mean = ndimage.uniform_filter(height_array, size=radius)
    local_variance = ndimage.uniform_filter(
        height_array ** 2, size=radius) - local_mean ** 2

    # Lower areas with high local variance = more occluded
    occlusion = 1.0 - np.clip(local_variance * intensity * 10.0, 0.0, 1.0)

    # Additional occlusion from slope (steeper = darker)
    grad_x = np.gradient(height_array, axis=1)
    grad_y = np.gradient(height_array, axis=0)
    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
    slope_ao = 1.0 - np.clip(slope * intensity * 2.0, 0.0, 0.5)

    # Combine
    ao = occlusion * slope_ao

    return np.clip(ao, 0.0, 1.0)


def apply_sharpening(
    img_array: np.ndarray,
    amount: float = 0.5,
    method: Literal["unsharp", "laplacian"] = "unsharp",
) -> np.ndarray:
    """
    Apply sharpening to enhance detail.

    Args:
        img_array: RGB image array (0-1 range)
        amount: Sharpening strength
        method: Sharpening algorithm

    Returns:
        Sharpened image
    """
    img = (img_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode="RGB")

    if method == "unsharp":
        # Unsharp mask
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=2))  # noqa: F841
        enhancer = ImageEnhance.Sharpness(pil_img)
        sharpened = enhancer.enhance(1.0 + amount)
        result = np.array(sharpened).astype(float) / 255.0

    elif method == "laplacian":
        # Laplacian sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * amount

        result = img_array.copy()
        if len(img_array.shape) == 3:
            for channel in range(img_array.shape[2]):
                result[:, :, channel] = ndimage.convolve(
                    img_array[:, :, channel], kernel, mode='reflect'
                )
        else:
            result = ndimage.convolve(img_array, kernel, mode='reflect')

    else:
        result = img_array

    return np.clip(result, 0.0, 1.0)


def apply_atmospheric_perspective(
    img_array: np.ndarray,
    height_array: np.ndarray,
    fog_color: tuple[float, float, float] = (0.7, 0.8, 0.9),
    fog_density: float = 0.3,
    fog_height_falloff: float = 2.0,
) -> np.ndarray:
    """
    Apply atmospheric perspective (fog/haze based on elevation).

    Args:
        img_array: RGB image array (0-1 range)
        height_array: Height values (0-1 range)
        fog_color: RGB fog/sky color
        fog_density: Overall fog density
        fog_height_falloff: How quickly fog dissipates with height

    Returns:
        Image with atmospheric perspective applied
    """
    # Calculate fog amount based on height (lower = more fog)
    fog_amount = fog_density * np.exp(-height_array * fog_height_falloff)
    fog_amount = np.clip(fog_amount, 0.0, 1.0)

    # Blend with fog color
    if len(img_array.shape) == 3:
        fog_amount = fog_amount[:, :, np.newaxis]
        # Ensure fog_rgb matches the number of channels
        if img_array.shape[2] == 4:
            fog_rgb = np.array([fog_color[0], fog_color[1], fog_color[2], 1.0])
        else:
            fog_rgb = np.array(fog_color)
    else:
        fog_rgb = np.array(fog_color)

    result = img_array * (1.0 - fog_amount) + fog_rgb * fog_amount

    return np.clip(result, 0.0, 1.0)
