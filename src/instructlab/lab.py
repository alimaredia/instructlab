# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-lines

# Standard
import logging
import multiprocessing
import os

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab import configuration as cfg
from instructlab import log

from instructlab.profiles.default_configuration import DefaultConfig
from instructlab.defaults import DEFAULTS

# 'fork' is unsafe and incompatible with some hardware accelerators.
# Python 3.14 will switch to 'spawn' on all platforms.
multiprocessing.set_start_method(cfg.DEFAULTS.MULTIPROCESSING_START_METHOD, force=True)

class Lab:
    """Lab object holds high-level information about ilab CLI"""
    def __init__(self, config_obj: DefaultConfig) -> None:
        self.config = config_obj

def ensure_storage_directories_exist(logger: logging.Logger) -> None:
	"""
	Ensures that the default directories used by ilab exist.
	"""
	dirs_to_make = [
		DEFAULTS._cache_home,
		DEFAULTS._config_dir,
		DEFAULTS._data_dir,
		DEFAULTS.CHATLOGS_DIR,
		DEFAULTS.CHECKPOINTS_DIR,
		DEFAULTS.OCI_DIR,
		DEFAULTS.DATASETS_DIR,
		DEFAULTS.EVAL_DATA_DIR,
		DEFAULTS.INTERNAL_DIR,
		DEFAULTS.MODELS_DIR,
		DEFAULTS.TAXONOMY_DIR,
		DEFAULTS.TRAIN_CONFIG_DIR,
		DEFAULTS.TRAIN_PROFILE_DIR,
		DEFAULTS.TRAIN_ADDITIONAL_OPTIONS_DIR,
		DEFAULTS.PHASED_DIR,
	]
	logger.debug(f"Creating ilab default directories")
	for dirpath in dirs_to_make:
		if not os.path.exists(dirpath):
			os.makedirs(dirpath, exist_ok=True)
			logger.debug(f"Created {dirpath}")
		else:
			logger.debug(f"{dirpath} already exists")

@click.group(
    cls=clickext.ExpandAliasesGroup,
    ep_group="instructlab.command",
    alias_ep_group="instructlab.command.alias",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(),
    default=cfg.DEFAULTS.CONFIG_FILE,
    show_default=True,
    help="Path to a configuration file.",
)
@click.option(
    "-v",
    "--verbose",
    "debug_level",
    count=True,
    default=0,
    show_default=False,
    help="Enable debug logging (repeat for even more verbosity)",
)
@click.version_option(package_name="instructlab")
@click.pass_context
# pylint: disable=redefined-outer-name
def ilab(ctx, config_file, debug_level: int = 0):
    """CLI for interacting with InstructLab."""
    log.init_logging(bool(debug_level))
    logger = logging.getLogger()
    ensure_storage_directories_exist(logger)
    config = DefaultConfig()
    click.secho(f"Set default configuration", fg="green")

    ctx.obj = Lab(config_obj=config)
    ctx.default_map = config.model_dump(warnings=False)

    log.configure_logging(
        log_level=ctx.obj.config.general.log_level.upper(),
        debug_level=ctx.obj.config.general.debug_level,
        fmt=ctx.obj.config.general.log_format,
    )
