# SPDX-License-Identifier: Apache-2.0

# Standard

# Third Party
import click

# First Party
from instructlab import clickext

@click.group(
    cls=clickext.LazyEntryPointGroup,
    ep_group="instructlab.command.system",
)
@click.pass_context
def system(ctx):
    """Command group for all system-related command calls"""
    ctx.default_map = ctx.parent.default_map.get('system')
