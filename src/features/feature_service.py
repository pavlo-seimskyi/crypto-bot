from typing import Any, Dict, List

from src.features.feature_generators.abstract import FeatureGenerator


class FeatureService:
    def __init__(self, *feature_generators: List[FeatureGenerator]):
        self.feature_generators = feature_generators

    def initialize(self, data: Any) -> None:
        for feature_generator in self.feature_generators:
            feature_generator.initialize(data)

    def add_value(self, data_row: Any, purging: bool = False):
        for feature_generator in self.feature_generators:
            feature_generator.add_value(data_row, purging)

    @property
    def output_values(self) -> Dict[str, List[float]]:
        return self.flatten_dict(
            {
                feature_generator.name: feature_generator.output_values
                for feature_generator in self.feature_generators
            }
        )

    @staticmethod
    def flatten_dict(input_dict) -> Dict[str, Any]:
        output = {}
        for k, v in input_dict.items():
            if isinstance(v, dict):
                for inner_key, inner_value in v.items():
                    output[f"{k}__{inner_key}"] = inner_value
            else:
                output[k] = v
        return output
