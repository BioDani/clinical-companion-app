"""BMI from a user message that includes weight and height."""

from __future__ import annotations

import re

_NUMBER = r"(\d+(?:[.,]\d+)?)"
_KG = re.compile(rf"{_NUMBER}\s*kg\b", re.IGNORECASE)
_CM = re.compile(rf"{_NUMBER}\s*cm\b", re.IGNORECASE)
_METERS = re.compile(rf"{_NUMBER}\s*m\b", re.IGNORECASE)
_KG_UNIT = re.compile(r"\bkg\b", re.IGNORECASE)
_CM_UNIT = re.compile(r"\bcm\b", re.IGNORECASE)
_BMI_ASK = re.compile(r"\bbmi\b|body\s+mass\s+index", re.IGNORECASE)

NEED_BOTH = (
    "To calculate your BMI, share your weight in kilograms and your height "
    "in centimeters or meters. This is informational guidance only."
)
NEED_WEIGHT = (
    "To calculate your BMI, I still need your weight in kilograms. "
    "This is informational guidance only."
)
NEED_HEIGHT = (
    "To calculate your BMI, I still need your height in centimeters or meters. "
    "This is informational guidance only."
)
INVALID_MEASUREMENTS = (
    "Those measurements cannot be used for a BMI reading. "
    "Share a weight in kilograms and a height in centimeters or meters. "
    "This is informational guidance only."
)


def _labeled_number(text: str, label: str) -> float | None:
    for line in (text or "").splitlines():
        if re.search(rf"\b{label}\b", line, re.IGNORECASE) is None:
            continue
        numbers = re.findall(r"\d+(?:[.,]\d+)?", line)
        if numbers:
            return float(numbers[-1].replace(",", "."))
    return None


def _number(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    return float(match.group(1).replace(",", "."))


class BmiTool:
    def from_text(self, question: str) -> str | None:
        weight_kg = _number(_KG.search(question))
        height_m = self._height_m(question)
        if weight_kg is None or height_m is None or weight_kg <= 0 or height_m <= 0:
            return None
        return _reading(weight_kg, height_m)

    def respond(self, question: str, remembered: str = "") -> str | None:
        """BMI reading, a request for the missing measurement, or None.

        A BMI request fills a missing measurement from remembered weight and height.
        A message that does not ask for BMI never reads that text.
        """
        weight_kg = _number(_KG.search(question))
        height_m = self._height_m(question)
        asks = _BMI_ASK.search(question or "") is not None
        if weight_kg is not None and weight_kg <= 0 and (asks or height_m is not None):
            return INVALID_MEASUREMENTS
        if height_m is not None and height_m <= 0 and (asks or weight_kg is not None):
            return INVALID_MEASUREMENTS
        if asks and _KG_UNIT.search(question or "") and weight_kg is None:
            return INVALID_MEASUREMENTS
        if asks and _CM_UNIT.search(question or "") and height_m is None:
            return INVALID_MEASUREMENTS
        if asks and remembered.strip():
            if weight_kg is None:
                weight_kg = self._memory_weight(remembered)
            if height_m is None:
                height_m = self._memory_height(remembered)
            if (weight_kg is not None and weight_kg <= 0) or (
                height_m is not None and height_m <= 0
            ):
                return INVALID_MEASUREMENTS
        if weight_kg is not None and height_m is not None:
            return _reading(weight_kg, height_m)
        if not asks:
            return None
        if weight_kg is None and height_m is None:
            return NEED_BOTH
        if weight_kg is None:
            return NEED_WEIGHT
        return NEED_HEIGHT

    def _memory_weight(self, text: str) -> float | None:
        found = _number(_KG.search(text))
        if found is not None:
            return found
        return _labeled_number(text, "weight")

    def _memory_height(self, text: str) -> float | None:
        found = self._height_m(text)
        if found is not None:
            return found
        value = _labeled_number(text, "height")
        if value is None:
            return None
        if value > 3:
            return value / 100
        return value

    def _height_m(self, question: str) -> float | None:
        centimeters = _number(_CM.search(question))
        if centimeters is not None:
            return centimeters / 100
        meters = _number(_METERS.search(question))
        if meters is None:
            return None
        return meters


def _reading(weight_kg: float, height_m: float) -> str:
    bmi = weight_kg / (height_m ** 2)
    return f"BMI {bmi:.1f} ({_category(bmi)})"


def _category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"
