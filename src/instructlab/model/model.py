# SPDX-License-Identifier: Apache-2.0

# pylint: disable=redefined-builtin
# Standard

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import storage_dirs_exist


@click.group(
    cls=clickext.LazyEntryPointGroup,
    ep_group="instructlab.command.model",
)
@click.pass_context
# pylint: disable=redefined-outer-name
def model(ctx):
    """Command Group for Interacting with the Models in InstructLab.
    """
    ctx.default_map = ctx.parent.default_map
