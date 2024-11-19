import os
import yaml
import logging
from math import floor
from typing import Optional
from pydantic import ValidationError
from instructlab.profiles.default_configuration import DefaultConfig
from instructlab.utils import convert_bytes_to_proper_mag
from deepmerge import always_merger, merge_or_raise

logger = logging.getLogger(__name__)

def autodetect_profile() -> tuple[Optional[str], Optional[str]]:
    # Third Party
    import torch
    import platform

    profile_file = None
    profile_name = None
    vram = 0
    gpus = 0
    machine = platform.machine()
    system = platform.system()
    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        total_vram = 0
        for i in range(gpus):
            properties = torch.cuda.get_device_properties(i)
            orig_chip_name = properties.name.lower()
            chip_name = orig_chip_name.replace(" ", "-")
            total_vram += properties.total_memory  # memory in B

        vram = int(floor(convert_bytes_to_proper_mag(total_vram)[0]))
        profile_name = f"{chip_name}x{gpus}"
        profile_file = f"{chip_name}x{gpus}.yaml"
        logger.debug(f"Auto-detected {gpus} {orig_chip_name} GPUs")
        logger.debug(f"Auto-detected {vram} GB of GPU VRAM")
    elif system == "Darwin" and machine == "arm64":
        profile_name = "Apple M-Series"
        profile_file = "apple-m-series.yaml"
        logger.debug(f"Auto-detected an Apple M-Series device")

    autodetect_profiles_dir = os.path.join(os.path.dirname(__file__), "autodetected")
    profile_file = os.path.join(autodetect_profiles_dir, profile_file)

    if profile_file is not None and os.path.isfile(profile_file):
        logger.debug(f"Auto-detected custom configuration {profile_name}")
        return profile_file, profile_name
    else:
        logger.debug(f"Did not auto-detect custom hardware configuration")
        if gpus == 1 and vram >= 16:
            profile_name = "nvidia-1xgpu-16gb"
            profile_file = os.path.join(autodetect_profiles_dir, f"{profile_name}.yaml")
        elif gpus == 4 and vram > 16:
            profile_name = "nvidia-4xgpu-16gb"
            profile_file = os.path.join(autodetect_profiles_dir, f"{profile_name}.yaml")
        else:
            profile_file = None
            profile_name = "default"

        if profile_file is not None:
            logger.debug(f"Auto-detected general configuration {profile_name}")
        else:
            logger.debug(f"Did not auto-detect any configuration for this hardware")

    return profile_file, profile_name

def load_profile(profile_file: str) -> dict:
	hardware_profile = {}
	try:
		with open(profile_file, 'r') as file:
			hardware_profile = yaml.safe_load(file)
	except OSError:
		logger.debug(f"Error: The file '{profile_file}' was not found.")

	return hardware_profile

def print_bad_keys(validation_error: ValidationError) -> None:
    errors = validation_error.errors(include_url=False)
    for error in errors:
        bad_key = ""
        for loc in error['loc']:
            if loc == error['loc'][-1]:
                bad_key += (loc)
            else:
                bad_key += (loc + ".")
        var_type = "section"
        if len(error['loc']) == 3:
            var_type = "variable"
        logger.debug(f"{bad_key} is not a valid configuration {var_type}")

def apply_profile(config: DefaultConfig, profile: dict) -> DefaultConfig:
	try:
		DefaultConfig.model_validate(profile)
	except ValidationError as e:
		print_bad_keys(e)
		return config

	config_dict = config.model_dump(warnings=False)

	# TODO look into using merge_or_raise instead
	#profile_dict = merge_or_raise.merge(config_dict, profile)
	profile_dict = always_merger.merge(config_dict, profile)

	try:
		new_config = DefaultConfig(**profile_dict)
	except ValidationError as e:
		print_bad_keys(e)
		return config

	return new_config
