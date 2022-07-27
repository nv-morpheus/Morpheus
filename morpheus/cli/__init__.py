import pluggy

from morpheus.cli.register_stage import register_stage

hookimpl = pluggy.HookimplMarker("morpheus")
"""Marker to be imported and used in plugins (and for own implementations)"""
