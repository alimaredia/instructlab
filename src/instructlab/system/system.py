# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext

logger = logging.getLogger(__name__)


@click.group(
    cls=clickext.LazyEntryPointGroup,
    ep_group="instructlab.command.system",
)
@click.pass_context
def system(ctx):
    """Command group for all system-related command calls"""
    #print(ctx.obj)
    #ctx.obj = ctx.parent.obj
    #ctx.ali = ctx.parent.ali
    #ctx.default_map = ctx.parent.default_map
    #ctx.ali = "hello it's ali in system() from ilab()"
    #print(ctx.ali)
