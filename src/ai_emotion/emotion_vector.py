from dataclasses import dataclass, fields
import numpy as np
import pandas as pd
from typing import Self


@dataclass
class EmotionVector:
    def to_series(self):
        return pd.Series(self.__dict__)

    @classmethod
    def from_series(cls, series: pd.Series) -> Self:
        # convert a series back to an instance of the class
        kwargs = {field.name: series[field.name] for field in fields(cls)}
        return cls(**kwargs)

    @classmethod
    def description(cls) -> str:
        return "Description of fields in emotion vector."

    def length(self):
        series = self.to_series()
        return np.linalg.norm(series)

    def normalize(self):
        series = self.to_series()
        normalized_series = series / self.length()
        return self.from_series(normalized_series)

    def subtract(self, other: Self) -> Self:
        if not isinstance(other, type(self)):
            raise ValueError("Subtraction requires another instance of the same class")
        series1 = self.to_series()
        series2 = other.to_series()
        return self.from_series(series1 - series2)

    def add(self, other: Self) -> Self:
        if not isinstance(other, type(self)):
            raise ValueError("Addition requires another instance of the same class")
        series1 = self.to_series()
        series2 = other.to_series()
        return self.from_series(series1 + series2)
