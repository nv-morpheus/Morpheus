
# Run the tool to get AWS credentials
source ./get_aws_credentials.sh

# Run the training forwarding any args
python dfp_pipeline_duo.py "$@"
