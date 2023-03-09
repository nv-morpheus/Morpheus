#!/bin/sh

export set FLASK_APP=webapp

THIS_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

APP_PATH="$THIS_DIR/../hil_app"

#$(cd $APP_PATH && python -m flask run)

# Run this command if default port is already being used.
$(cd $APP_PATH && python -m flask run -p 3000)
