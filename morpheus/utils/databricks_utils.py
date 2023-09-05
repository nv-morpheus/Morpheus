import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def configure_databricks_connect(databricks_host: str,
                                  databricks_token: str,
                                  databricks_cluster_id: str,
                                  databricks_port: str,
                                  databricks_org_id: str):
    if (os.environ.get('DATABRICKS_HOST', None) is None and databricks_host is None):
        raise Exception("Parameter for databricks host not provided")
    if (os.environ.get('DATABRICKS_TOKEN', None) is None and databricks_token is None):
        raise Exception("Parameter for databricks token not provided")
    if (os.environ.get('DATABRICKS_CLUSTER_ID', None) is None and databricks_cluster_id is None):
        raise Exception("Parameter for databricks cluster not provided")

    config_file = f"{str(Path.home())}/.databricks-connect"
    add_config = True

    config = {
        "host": os.environ.get('DATABRICKS_HOST', databricks_host),
        "token": os.environ.get('DATABRICKS_TOKEN', databricks_token),
        "cluster_id": os.environ.get('DATABRICKS_CLUSTER_ID', databricks_cluster_id),
        "org_id": databricks_org_id,
        "port": databricks_port
    }
    # check if the config file for databricks connect already exists
    config = json.dumps(config)
    if os.path.exists(config_file):
        # check if the config being added already exists, if so do nothing
        with open(config_file) as f:
            if config in f.read():
                logger.info("Configuration for databricks-connect already exists, nothing added!")
                add_config = False
            else:
                logger.info("Configuration not found for databricks-connect, adding provided configs!")
                
    if add_config:
        with open(config_file, "w+") as f:
            f.write(f"\n{config}\n")
        logger.info("Databricks-connect successfully configured!")
