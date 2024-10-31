# SPDX-License-Identifier: Apache-2.0
# Standard

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import storage_dirs_exist


@click.group(
    cls=clickext.LazyEntryPointGroup,
    ep_group="instructlab.command.config",
)
@click.pass_context
def config(ctx):
    """Command Group for Interacting with the Config of InstructLab."""
    ctx.config = ctx.parent.config
