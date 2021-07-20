#!/bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#########################
# Morpheus Style Tester #
#########################

# Ignore errors and set path
set +e
PATH=/opt/conda/bin:$PATH
LC_ALL=C.UTF-8
LANG=C.UTF-8

# Activate common conda env if a conda env isnt already activated
if [ "$CONDA_PREFIX" = "" ]; then
  . /opt/conda/etc/profile.d/conda.sh
  conda activate rapids
fi

# Run isort and get results/return code
ISORT=`isort --check-only morpheus`
ISORT_RETVAL=$?

# Run flake8 and get results/return code
FLAKE=`flake8`
FLAKE_RETVAL=$?

# Check for copyright headers in the files modified currently
COPYRIGHT=`python ci/checks/copyright.py --git-modified-only 2>&1`
COPYRIGHT_RETVAL=$?

# Output results if failure otherwise show pass
if [ "$ISORT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: isort style check; begin output\n\n"
  echo -e "$ISORT"
  echo -e "\n\n" \
          ">>>> FAILED: isort style check; end output\n" \
          "             run 'isort morpheus' to auto-fix\n\n"
else
  echo -e "\n\n>>>> PASSED: isort style check\n\n"
fi

if [ "$FLAKE_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: flake8 style check; begin output\n\n"
  echo -e "$FLAKE"
  echo -e "\n\n>>>> FAILED: flake8 style check; end output\n\n"
  echo -e "\n\n" \
          ">>>> FAILED: flake8 style check; end output; end output\n" \
          "             run 'yapf -i -r morpheus' to auto-fix many issues (not all)\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8 style check\n\n"
fi

if [ "$COPYRIGHT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: copyright check; begin output\n\n"
  echo -e "$COPYRIGHT"
  echo -e "\n\n>>>> FAILED: copyright check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: copyright check\n\n"
fi

RETVALS=($ISORT_RETVAL $FLAKE_RETVAL $COPYRIGHT_RETVAL)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
