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
    source = np.asarray([24.0 / 24.0, 273.15, GRAVITY_M_S2 * 10.0])
    converted = convert_fields(source[:, None, None])
    assert converted["precipitation"].item() == 24.0
    assert converted["temperature"].item() == 0.0
    assert converted["z500"].item() == 1.0


def test_global_quantization_round_trip() -> None:
    examples = {
        "precipitation": np.asarray([0.0, 7.25, 123.47]),
        "temperature": np.asarray([-82.12, 0.0, 42.37]),
        "z500": np.asarray([412.31, 553.27, 618.91]),
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
    source_spread = np.asarray([1.0, 2.5, GRAVITY_M_S2 * 10.0])
    converted = convert_spread(source_spread[:, None, None])
    assert converted["precipitation"].item() == 24.0
    assert converted["temperature"].item() == 2.5
    assert converted["z500"].item() == 1.0
