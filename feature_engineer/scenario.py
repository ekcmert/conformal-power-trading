from __future__ import annotations

import pandas as pd

from .common import coerce_numeric_frame


def engineer_scenario_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    features = coerce_numeric_frame(dataframe)

    range_features: dict[str, pd.Series] = {}
    for column in features.columns:
        if not column.endswith("__max"):
            continue

        base_name = column[: -len("__max")]
        min_column = f"{base_name}__min"
        if min_column not in features.columns:
            continue

        range_features[f"{base_name}__range"] = features[column] - features[min_column]

    if range_features:
        features = pd.concat([features, pd.DataFrame(range_features, index=features.index)], axis=1)

    return features


__all__ = ["engineer_scenario_features"]
