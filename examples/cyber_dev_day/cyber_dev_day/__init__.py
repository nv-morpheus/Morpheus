import logging
import sys
from morpheus.utils.logger import configure_logging

morpheus_logger = logging.getLogger("morpheus")

# if (not getattr(morpheus_logger, "_configured_by_morpheus", False)):

#     # Configure logging to just be INFO level at first. This can be changed later. Need the Morpheus logger to be
#     # configured first
#     configure_logging(log_level=logging.INFO)

# Set the morpheus logger to propagate upstream
morpheus_logger.propagate = True

logger = logging.getLogger(__name__)

# Set the parent logger for the entire package to use morpheus so we can take advantage of configure_logging
logger.parent = morpheus_logger

# Finally, get the root logger and add a default handler to it to print to screen
root_logger = logging.getLogger()

root_logger.addHandler(logging.StreamHandler(stream=sys.stdout))
