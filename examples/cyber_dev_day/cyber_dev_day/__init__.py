import logging
import sys

morpheus_logger = logging.getLogger("morpheus")

if (not getattr(morpheus_logger, "_configured_by_morpheus", False)):

    # Set the morpheus logger to propagate upstream
    morpheus_logger.propagate = False

    # Add a default handler to the morpheus logger to print to screen
    morpheus_logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    # Set a flag to indicate that the logger has been configured by Morpheus
    setattr(morpheus_logger, "_configured_by_morpheus", True)

logger = logging.getLogger(__name__)

# Set the parent logger for the entire package to use morpheus so we can take advantage of configure_logging
logger.parent = morpheus_logger
