import numpy as np
import numpy.typing as npt

from enum import Enum

from math import log2, ceil

import os
c_sources_abspath = f"{os.path.dirname(__file__)}/sources_c"

def panic():
    print("PANIC!")
    exit(1)

class CTypes(Enum):
    u8  = 0
    i8  = 1
    u16 = 2
    i16 = 3
    u32 = 4
    i32 = 5
    f32 = 6
    f64 = 7

CTypes_data_ranges = {
    CTypes.u8:  [0,        2**8 - 1],
    CTypes.i8:  [-(2**7),  2**7 - 1],
    CTypes.u16: [0,        2**16 - 1],
    CTypes.i16: [-(2**15), 2**15 - 1],
    CTypes.u32: [0,        2**32 - 1],
    CTypes.i32: [-(2**31), 2**31 - 1]
}

CTypes_size_bytes = {
    CTypes.u8:  1,
    CTypes.i8:  1,
    CTypes.u16: 2,
    CTypes.i16: 2,
    CTypes.u32: 4,
    CTypes.i32: 4,
    CTypes.f32: 4,
    CTypes.f64: 8
}

def find_ctype(signed: bool, max_val: int) -> str:
    bits = ceil(log2(max_val))
    dt = ""

    if signed:
        bits += 1

    if bits <= 8:
        dt = "8"
    elif bits <= 16:
        dt = "16"
    else:
        dt = "32"

    if signed:
        dt = "i" + dt
    else:
        dt = "u" + dt

    return CTypes[dt]

def saturate_to_ctype(x: npt.NDArray, data_type) -> npt.NDArray:
    if data_type in [CTypes.f32, CTypes.f64]:
        return x
    else:
        drange = CTypes_data_ranges[data_type]
        return np.clip(x, drange[0], drange[1]).astype(int)

def to_fixed(arr: npt.NDArray, fbits: int) -> npt.NDArray:
    return (arr * (2**fbits)).astype(int)

def update_data_range(prev_range: tuple[float,float], arr: npt.NDArray) -> tuple[float,float]:
    if prev_range is None:
        return (np.min(arr), np.max(arr))
    else:
        min_v, max_v = prev_range
        return (min(min_v, np.min(arr)), max(max_v, np.max(arr)))

def humanize_bytes(b: int):
    i = 0
    while b >= 1024 and i < 3:
        b //= 1024
        i += 1
    return f"{b} {["B", "KiB", "MiB", "GiB"][i]}"

class Distributions(Enum):
    gaussian = 0
    uniform = 1
    bernoulli = 2

class InferenceMode(Enum):
    sampling = 0
    distribution = 1

    def __str__(self):
        if self == InferenceMode.sampling:
            return "Sampling"
        else:
            return "Moment Prop"

def uniform_transform(mu, sigma):
    b = sigma * np.sqrt(12.0)
    a = mu - (b / 2)
    return (a, b)

def bernoulli_transform(mu, sigma):
    mu += np.finfo(np.float64).eps
    p = mu**2 / (mu**2 + sigma**2)
    q = (mu**2 + sigma**2) / mu
    return (p, q)
