from typing import Any, Dict, TypedDict

from torchao.core.config import AOBaseConfig
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Float8StaticActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
)


# define a TypedDict class for storing PTQ configurations and associated metadata
class ConfigAndMetadataPTQ(TypedDict, total=True):
    ao_config: type[AOBaseConfig]
    precision: str
    bits_per_weight: int
    dynamic_calibration: bool
    weight_only: bool
    cuda_compute_capacity_compatibility: float
    ao_config_kwargs: Dict[str, Any]


PTQ_WEIGHT_ONLY_CONFIG_METADATA = {
    "Int8WeightOnlyConfig": ConfigAndMetadataPTQ(
        ao_config=Int8WeightOnlyConfig,
        precision="int8",
        bits_per_weight=8,
        dynamic_calibration=False,
        weight_only=True,
        cuda_compute_capacity_compatibility=8.6,
        ao_config_kwargs={"version": 2},
    ),
    "Float8WeightOnlyConfig": ConfigAndMetadataPTQ(
        ao_config=Float8WeightOnlyConfig,
        precision="float8",
        bits_per_weight=8,
        dynamic_calibration=False,
        weight_only=True,
        cuda_compute_capacity_compatibility=8.6,
        ao_config_kwargs={"version": 2},
    ),
    "Int4WeightOnlyConfig": ConfigAndMetadataPTQ(
        ao_config=Int4WeightOnlyConfig,
        precision="int4",
        bits_per_weight=4,
        dynamic_calibration=False,
        weight_only=True,
        cuda_compute_capacity_compatibility=8.6,
        ao_config_kwargs={"version": 2},
    ),
}

PTQ_WEIGHT_ACTIVATION_CONFIG_METADATA = {
    "Float8DynamicActivationFloat8WeightConfig": ConfigAndMetadataPTQ(
        ao_config=Float8DynamicActivationFloat8WeightConfig,
        precision="float8",
        bits_per_weight=8,
        dynamic_calibration=True,
        weight_only=False,
        cuda_compute_capacity_compatibility=8.6,
        ao_config_kwargs={"version": 2},
    ),
    "Float8StaticActivationFloat8WeightConfig": ConfigAndMetadataPTQ(
        ao_config=Float8StaticActivationFloat8WeightConfig,
        precision="float8",
        bits_per_weight=8,
        dynamic_calibration=False,
        weight_only=False,
        cuda_compute_capacity_compatibility=8.9,
        ao_config_kwargs={},
    ),
    "Int8DynamicActivationInt8WeightConfig": ConfigAndMetadataPTQ(
        ao_config=Int8DynamicActivationInt8WeightConfig,
        precision="int8",
        bits_per_weight=8,
        dynamic_calibration=True,
        weight_only=False,
        cuda_compute_capacity_compatibility=8.6,
        ao_config_kwargs={"version": 2},
    ),
    "Float8DynamicActivationInt4WeightConfig": ConfigAndMetadataPTQ(
        ao_config=Float8DynamicActivationInt4WeightConfig,
        precision="float8act_int4weight",
        bits_per_weight=4,
        dynamic_calibration=True,
        weight_only=False,
        cuda_compute_capacity_compatibility=8.6,
        ao_config_kwargs={},
    ),
}

PTQ_QUANT_CONFIG_METADATA = (
    PTQ_WEIGHT_ONLY_CONFIG_METADATA | PTQ_WEIGHT_ACTIVATION_CONFIG_METADATA
)
