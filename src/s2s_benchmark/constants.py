"""Shared constants for the benchmark analysis."""

from __future__ import annotations

from dataclasses import dataclass


VARIABLES = ("TP", "Z500", "T2M")

MODEL_VARIABLES = {
    "ECMWF": ("TP", "Z500", "T2M"),
    "FuXi": ("TP", "Z500", "T2M"),
    "UKMO": ("TP", "Z500", "T2M"),
    "DELYSM": ("Z500", "T2M"),
}

VARIABLE_LABELS = {
    "TP": "Precipitation",
    "Z500": "500 hPa geopotential height",
    "T2M": "2 m temperature",
}

VARIABLE_UNITS = {
    "TP": "mm day-1",
    "Z500": "gpm",
    "T2M": "K",
}


@dataclass(frozen=True)
class LeadWindow:
    """Inclusive 1-based lead-day window."""

    name: str
    start: int
    end: int

    @property
    def n_days(self) -> int:
        return self.end - self.start + 1

    @property
    def week(self) -> int:
        return int(self.name.replace("Week ", ""))


WEEKLY_WINDOWS = (
    LeadWindow("Week 1", 1, 7),
    LeadWindow("Week 2", 8, 14),
    LeadWindow("Week 3", 15, 21),
    LeadWindow("Week 4", 22, 28),
    LeadWindow("Week 5", 29, 35),
    LeadWindow("Week 6", 36, 42),
)


def get_week(name_or_number: str | int) -> LeadWindow:
    """Return one standard weekly lead window."""

    if isinstance(name_or_number, int):
        target = f"Week {name_or_number}"
    else:
        text = str(name_or_number).strip()
        target = text if text.lower().startswith("week") else f"Week {int(text)}"
    for window in WEEKLY_WINDOWS:
        if window.name == target:
            return window
    raise ValueError(f"unknown lead window: {name_or_number!r}")
