import logging

morpheus_logger = logging.getLogger("morpheus")

logger = logging.getLogger(__name__)

# Set the parent logger for the entire package to use morpheus so we can take advantage of configure_logging
logger.parent = morpheus_logger
