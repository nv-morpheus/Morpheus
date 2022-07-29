import pluggy

# Must be before the other imports
hookimpl = pluggy.HookimplMarker("morpheus")
"""Marker to be imported and used in plugins (and for own implementations)"""

from morpheus.cli.register_stage import register_stage
from morpheus.cli.run import run_cli  # Import the run_cli command into the cli module
