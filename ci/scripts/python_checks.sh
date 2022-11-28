#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Based on style.sh from Morpheus

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/common.sh

# Ignore errors and set path
set +e
LC_ALL=C.UTF-8
LANG=C.UTF-8

# Pre-populate the return values in case they are skipped
ISORT_RETVAL=0
FLAKE_RETVAL=0
YAPF_RETVAL=0

get_modified_files ${PYTHON_FILE_REGEX} MORPHEUS_MODIFIED_FILES

# When invoked by the git pre-commit hook CHANGED_FILES will already be defined
if [[ -n "${MORPHEUS_MODIFIED_FILES}" ]]; then
   echo -e "Running Python checks on ${#MORPHEUS_MODIFIED_FILES[@]} files:"

   for f in "${MORPHEUS_MODIFIED_FILES[@]}"; do
      echo "  $f"
   done

   if [[ "${SKIP_ISORT}" == "" ]]; then
      # Run using a clang-tidy wrapper to allow warnings-as-errors and to eliminate any output except errors (since clang-tidy-diff.py doesn't return the correct error codes)
      ISORT_OUTPUT=`python3 -m isort --settings-file ${PY_CFG} --filter-files --check-only  ${MORPHEUS_MODIFIED_FILES[@]} 2>&1`
      ISORT_RETVAL=$?
   fi

   if [[ "${SKIP_FLAKE}" == "" ]]; then
      # Run using a clang-tidy wrapper to allow warnings-as-errors and to eliminate any output except errors (since clang-tidy-diff.py doesn't return the correct error codes)
      FLAKE_OUTPUT=`python3 -m flake8 --config ${PY_CFG} ${MORPHEUS_MODIFIED_FILES[@]} 2>&1`
      FLAKE_RETVAL=$?
   fi

   if [[ "${SKIP_YAPF}" == "" ]]; then
      # Run yapf. Will return 1 if there are any diffs
      YAPF_OUTPUT=`python3 -m yapf --style ${PY_CFG} --diff ${MORPHEUS_MODIFIED_FILES[@]} 2>&1`
      YAPF_RETVAL=$?
   fi

else
   echo "No modified Python files to check"
fi

# Output results if failure otherwise show pass
if [[ "${SKIP_ISORT}" != "" ]]; then
   echo -e "\n\n>>>> SKIPPED: isort check\n\n"
elif [ "${ISORT_RETVAL}" != "0" ]; then
   echo -e "\n\n>>>> FAILED: isort style check; begin output\n\n"
   echo -e "${ISORT_OUTPUT}"
   echo -e "\n\n>>>> FAILED: isort style check; end output\n\n" \
           "To auto-fix many issues (not all) run:\n" \
           "   ./ci/scripts/fix_all.sh\n\n"
else
  echo -e "\n\n>>>> PASSED: isort style check\n\n"
fi

if [[ "${SKIP_FLAKE}" != "" ]]; then
   echo -e "\n\n>>>> SKIPPED: flake8 check\n\n"
elif [ "${FLAKE_RETVAL}" != "0" ]; then
   echo -e "\n\n>>>> FAILED: flake8 style check; begin output\n\n"
   echo -e "${FLAKE_OUTPUT}"
   echo -e "\n\n>>>> FAILED: flake8 style check; end output\n\n" \
           "To auto-fix many issues (not all) run:\n" \
           "   ./ci/scripts/fix_all.sh\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8 style check\n\n"
fi

if [[ "${SKIP_YAPF}" != "" ]]; then
   echo -e "\n\n>>>> SKIPPED: yapf check\n\n"
elif [ "${YAPF_RETVAL}" != "0" ]; then
   echo -e "\n\n>>>> FAILED: yapf style check; begin output\n\n"
   echo -e "Incorrectly formatted files:"
   YAPF_OUTPUT=`echo "${YAPF_OUTPUT}" | sed -nr 's/^\+\+\+ ([^ ]*) *\(reformatted\)$/\1/p'`
   echo -e "${YAPF_OUTPUT}"
   echo -e "\n\n>>>> FAILED: yapf style check; end output\n\n" \
           "To auto-fix many issues (not all) run:\n" \
           "   ./ci/scripts/fix_all.sh\n\n"
else
  echo -e "\n\n>>>> PASSED: yapf style check\n\n"
fi

RETVALS=(${ISORT_RETVAL} ${FLAKE_RETVAL})
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
