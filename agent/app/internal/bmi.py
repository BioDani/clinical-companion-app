"""BMI from a user message that includes weight and height."""

from __future__ import annotations

import re

_NUMBER = r"(\d+(?:[.,]\d+)?)"
_KG = re.compile(rf"{_NUMBER}\s*kg\b", re.IGNORECASE)
_CM = re.compile(rf"{_NUMBER}\s*cm\b", re.IGNORECASE)
_METERS = re.compile(rf"{_NUMBER}\s*m\b", re.IGNORECASE)


def _number(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    return float(match.group(1).replace(",", "."))


class BmiTool:
    def from_text(self, question: str) -> str | None:
        weight_kg = _number(_KG.search(question))
        height_m = self._height_m(question)
        if weight_kg is None or height_m is None or height_m <= 0:
            return None
        bmi = weight_kg / (height_m ** 2)
        return f"BMI {bmi:.1f} ({_category(bmi)})"

    def _height_m(self, question: str) -> float | None:
        centimeters = _number(_CM.search(question))
        if centimeters is not None:
            return centimeters / 100
        meters = _number(_METERS.search(question))
        if meters is None:
            return None
        return meters


def _category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"
