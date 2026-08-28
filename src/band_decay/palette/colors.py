"""Color-space conversion and color-vision simulation helpers."""

from __future__ import annotations

import numpy as np


_CVD_MATRICES = {
    "protanopia": np.asarray(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ],
        dtype=float,
    ),
    "deuteranopia": np.asarray(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ],
        dtype=float,
    ),
}


def _candidate_labs_for_conditions(candidate_srgb: np.ndarray, conditions: tuple[str, ...]) -> np.ndarray:
    """Return OKLab coordinates for normal and simulated color-vision states."""

    # Evaluate every configured vision condition before comparing candidates.
    labs = []
    ordered_conditions = ("normal", *(condition for condition in conditions if condition != "normal"))
    for condition in ordered_conditions:
        if condition == "normal":
            condition_srgb = candidate_srgb
        else:
            condition_srgb = _simulate_cvd_srgb(candidate_srgb, condition)
        labs.append(_srgb_to_oklab(condition_srgb))
    return np.stack(labs, axis=0)


def _simulate_cvd_srgb(srgb: np.ndarray, condition: str) -> np.ndarray:
    """Simulate full protanopia or deuteranopia in linear sRGB."""

    # Simulate in linear light, then encode the result back to sRGB.
    transformed = _srgb_to_linear(srgb) @ _CVD_MATRICES[condition].T
    return _linear_to_srgb(np.clip(transformed, 0.0, 1.0))


def _oklch_to_oklab(lightness: np.ndarray, chroma: np.ndarray, hue_degrees: np.ndarray) -> np.ndarray:
    hue_radians = np.deg2rad(hue_degrees)
    return np.stack(
        [lightness, chroma * np.cos(hue_radians), chroma * np.sin(hue_radians)],
        axis=-1,
    )


def _oklab_to_oklch(oklab: np.ndarray) -> np.ndarray:
    oklab = np.asarray(oklab, dtype=float)
    return np.stack(
        [
            oklab[..., 0],
            np.hypot(oklab[..., 1], oklab[..., 2]),
            np.mod(np.rad2deg(np.arctan2(oklab[..., 2], oklab[..., 1])), 360.0),
        ],
        axis=-1,
    )


def _oklab_to_srgb(oklab: np.ndarray) -> np.ndarray:
    # Apply the OKLab-to-sRGB matrix followed by the sRGB transfer curve.
    lightness = oklab[..., 0]
    axis_a = oklab[..., 1]
    axis_b = oklab[..., 2]
    l_prime = lightness + 0.3963377774 * axis_a + 0.2158037573 * axis_b
    m_prime = lightness - 0.1055613458 * axis_a - 0.0638541728 * axis_b
    s_prime = lightness - 0.0894841775 * axis_a - 1.2914855480 * axis_b
    lms = np.stack([l_prime ** 3, m_prime ** 3, s_prime ** 3], axis=-1)
    linear_srgb = np.stack(
        [
            4.0767416621 * lms[..., 0] - 3.3077115913 * lms[..., 1] + 0.2309699292 * lms[..., 2],
            -1.2684380046 * lms[..., 0] + 2.6097574011 * lms[..., 1] - 0.3413193965 * lms[..., 2],
            -0.0041960863 * lms[..., 0] - 0.7034186147 * lms[..., 1] + 1.7076147010 * lms[..., 2],
        ],
        axis=-1,
    )
    return _linear_to_srgb(linear_srgb)


def _srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
    # Decode sRGB and transform through LMS into perceptual OKLab coordinates.
    linear_srgb = _srgb_to_linear(np.asarray(srgb, dtype=float))
    lms = np.stack(
        [
            0.4122214708 * linear_srgb[..., 0]
            + 0.5363325363 * linear_srgb[..., 1]
            + 0.0514459929 * linear_srgb[..., 2],
            0.2119034982 * linear_srgb[..., 0]
            + 0.6806995451 * linear_srgb[..., 1]
            + 0.1073969566 * linear_srgb[..., 2],
            0.0883024619 * linear_srgb[..., 0]
            + 0.2817188376 * linear_srgb[..., 1]
            + 0.6299787005 * linear_srgb[..., 2],
        ],
        axis=-1,
    )
    lms_root = np.cbrt(np.clip(lms, 0.0, None))
    return np.stack(
        [
            0.2104542553 * lms_root[..., 0]
            + 0.7936177850 * lms_root[..., 1]
            - 0.0040720468 * lms_root[..., 2],
            1.9779984951 * lms_root[..., 0]
            - 2.4285922050 * lms_root[..., 1]
            + 0.4505937099 * lms_root[..., 2],
            0.0259040371 * lms_root[..., 0]
            + 0.7827717662 * lms_root[..., 1]
            - 0.8086757660 * lms_root[..., 2],
        ],
        axis=-1,
    )


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb = np.asarray(srgb, dtype=float)
    linear_srgb = np.empty_like(srgb, dtype=float)
    low_values = srgb <= 0.04045
    linear_srgb[low_values] = srgb[low_values] / 12.92
    linear_srgb[~low_values] = ((srgb[~low_values] + 0.055) / 1.055) ** 2.4
    return linear_srgb


def _linear_to_srgb(linear_srgb: np.ndarray) -> np.ndarray:
    linear_srgb = np.asarray(linear_srgb, dtype=float)
    srgb = np.empty_like(linear_srgb, dtype=float)
    low_values = linear_srgb <= 0.0031308
    srgb[low_values] = 12.92 * linear_srgb[low_values]
    srgb[~low_values] = 1.055 * np.maximum(linear_srgb[~low_values], 0.0) ** (1.0 / 2.4) - 0.055
    return srgb


def _relative_luminance(srgb: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(np.clip(np.asarray(srgb, dtype=float), 0.0, 1.0))
    return (
            0.2126 * linear[..., 0]
            + 0.7152 * linear[..., 1]
            + 0.0722 * linear[..., 2]
    )


def _srgb_to_hex(srgb: np.ndarray) -> str:
    rgb_uint8 = np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return f"#{rgb_uint8[0]:02x}{rgb_uint8[1]:02x}{rgb_uint8[2]:02x}"


def _hex_to_srgb(color: str) -> np.ndarray:
    value = str(color).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(character * 2 for character in value)
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit hexadecimal color, got {color!r}.")
    try:
        return np.asarray([int(value[index: index + 2], 16) / 255.0 for index in (0, 2, 4)], dtype=float)
    except ValueError as error:
        raise ValueError(f"Invalid hexadecimal color {color!r}.") from error
