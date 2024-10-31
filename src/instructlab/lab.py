# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-lines

# Standard
import multiprocessing
import os

# Third Party
import click
import yaml
from deepmerge import always_merger

# First Party
from instructlab import clickext
from instructlab import configuration as cfg
from .profiles.temp_default_profile import DefaultConfig

# 'fork' is unsafe and incompatible with some hardware accelerators.
# Python 3.14 will switch to 'spawn' on all platforms.
multiprocessing.set_start_method(cfg.DEFAULTS.MULTIPROCESSING_START_METHOD, force=True)


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
    """CLI for interacting with InstructLab.

    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    # this command shoud:
    # 1. run ensure_storage_directories_exist()
    # 2. read in the default configuration, then the do the autodetection, then 
    # 3. maybe propogate the --log-level, and --config options to every leaf node command

    # ctx.obj, what is it, it's a class Lab in confiugration.py
    # what I need is just a new Config object that has all of the values from the default config

    # what I want is the default config to be continuously overriden instead of the autodetection just picking up the right profile at the end of the day
    # but why?
        # because the overriding code would be easier when the config file is provided if the others were doing that too?
    config = DefaultConfig()

    autodetected_profiles_dir = os.path.join(os.path.dirname(__file__), "profiles/autodetected")
    if os.path.isdir(autodetected_profiles_dir):
        file_path = os.path.join(autodetected_profiles_dir, "96gb-vram.yaml")
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        print(f"autodetected profile config is: {yaml_data}")

        config = DefaultConfig(**yaml_data)
        print("Applied 96gb VRAM profile")
    else:
        print("skipping application of gpu autodetect profiles")

    custom_profiles_dir = os.path.join(os.path.dirname(__file__), "profiles/custom")
    if os.path.isdir(custom_profiles_dir):
        file_path = os.path.join(custom_profiles_dir, "l4-x4.yaml")
        # TODO: try block here
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        print(f"custom profile config is: {yaml_data}")

        # TODO: try block here
        DefaultConfig.model_validate(yaml_data)
        config_dict = config.model_dump()
        profile_dict = always_merger.merge(config_dict, yaml_data)
        config = config.model_copy(update=profile_dict)
        print("Applied l4x4 custom profile")
    else:
        print("skipping application of custom profiles")

    ctx.config = config

    #cfg.init(ctx, config_file, debug_level)
