from dataclasses import dataclass
from enum import IntEnum


class ProcessingStage(IntEnum):
    """Processing stages to determine what needs recalculation."""

    CALIBRATION = 0
    RAW_IRRADIANCE = 1
    SHADING = 2
    STATE_OF_CHARGE = 3
    NO_CHANGES = 4


@dataclass
class ModelConfig:
    """Configuration for sky image processing models."""

    model_name: str
    use_lgbm: bool
    resize_target: tuple


@dataclass
class DataSourceInfo:
    """Information about data sources used."""

    using_nasa: bool = False
    using_day_night: bool = False
    last_irradiance_file: str = ""


# System validation parameters
SYSTEM_PARAMS = {
    "LOCATION_PARAMS": ["Lattitude (°)", "Longitude (°)", "Elevation (m)"],
    "IMAGE_PARAMS": [
        "Image orientation (°)",
        "Image inclination (°)",
        "Plane orientation (°)",
        "Plane inclination (°)",
    ],
    "CALIB_PARAMS": ["Calib vertex short", "Calib vertex long", "Calib square size (mm)"],
    "TIME_PARAMS": ["Start year", "End year"],
    "SYSTEM_PARAMS": [
        "Solar panel peak wattage (W)",
        "Converter efficiency (%)",
        "Converter max power (W)",
        "Charge efficiency (%)",
        "Discharge efficiency (%)",
        "Max SOC (%)",
        "Min SOC (%)",
        "Batt nominal capacity (Ah)",
        "Batt nominal voltage (V)",
    ],
}

# Model configurations
MODEL_CONFIGS = {
    "1": ModelConfig("efficientnet-b5", False, (512, 512)),
    "2": ModelConfig("efficientnet-b7", False, (512, 512)),
    "3": ModelConfig("efficientnet-b5", False, (1024, 1024)),
    "4": ModelConfig("efficientnet-b7", False, (1024, 1024)),
    "5": ModelConfig("efficientnet-b5", True, (1024, 1024)),
    "6": ModelConfig("efficientnet-b7", True, (1024, 1024)),
}

# File paths
PATHS = {
    "system_data": "./SystemData",
    "debug_data": "./DebugData",
    "sky_images": "./SkyImageOfSite",
    "calibration_images": "./CalibrationImages",
    "system_specs": "./SystemData/System_Specifications.xlsx",
    "consumption_profile": "./SystemData/Consumption_Profile.xlsx",
    "day_night_profile": "./SystemData/Day_Night_Profile.xlsx",
    "status_file": "status.yml",
    "calibration_file": "calibration.yml",
    "irradiance_csv": "./DebugData/irradiance.csv",
}

# Validation limits
VALIDATION_LIMITS = {
    "latitude": (-90, 90),
    "longitude": (-180, 180),
    "elevation": (-500, 9000),
    "min_calib_images": 8,
    "angle_range": (0, 360),
    "percentage_range": (0, 100),
}
