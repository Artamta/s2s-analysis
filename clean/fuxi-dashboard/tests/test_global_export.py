from __future__ import annotations

import numpy as np

from scripts.export_global_forecast import (
    GRAVITY_M_S2,
    VARIABLES,
    convert_fields,
    convert_spread,
    quantize,
)


def test_global_unit_conversions_are_locked() -> None:
    source = np.asarray(
        [
            1.0,
            273.15,
            GRAVITY_M_S2 * 10.0,
            3.0,
            4.0,
            101_300.0,
            300.15,
            -240.0,
        ]
    )
    converted = convert_fields(source[:, None, None])
    assert converted["precipitation"].item() == 24.0
    assert converted["temperature"].item() == 0.0
    assert converted["z500"].item() == 1.0
    assert converted["wind850"].item() == 5.0
    assert converted["mslp"].item() == 1013.0
    assert converted["sst"].item() == 27.0
    assert converted["olr"].item() == 240.0


def test_global_quantization_round_trip() -> None:
    examples = {
        "precipitation": np.asarray([0.0, 7.25, 123.47]),
        "temperature": np.asarray([-82.12, 0.0, 42.37]),
        "z500": np.asarray([412.31, 553.27, 618.91]),
        "wind850": np.asarray([0.0, 12.25, 42.37]),
        "mslp": np.asarray([925.12, 1013.27, 1052.91]),
        "sst": np.asarray([-2.12, 20.0, 36.37]),
        "olr": np.asarray([18.12, 240.0, 356.37]),
    }
    for variable in VARIABLES:
        encoded = quantize(examples[variable.key], variable)
        decoded = encoded.astype(np.float64) * variable.scale + variable.offset
        np.testing.assert_allclose(
            decoded,
            examples[variable.key],
            atol=variable.scale / 2,
            rtol=0,
        )


def test_global_spread_conversion_has_no_absolute_offset() -> None:
    source_spread = np.asarray(
        [1.0, 2.5, GRAVITY_M_S2 * 10.0, 3.0, 4.0, 100.0, 2.0, 10.0]
    )
    converted = convert_spread(source_spread[:, None, None])
    assert converted["precipitation"].item() == 24.0
    assert converted["temperature"].item() == 2.5
    assert converted["z500"].item() == 1.0
    assert converted["mslp"].item() == 1.0
    assert converted["sst"].item() == 2.0
    assert converted["olr"].item() == 10.0
