#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
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
#

import argparse
import datetime
import io
import logging
import os
import re
import sys
import typing

# Now import gitutils. Ignore flake8 error here since there is no other way to
# set up imports
import gitutils  # noqa: E402

FilesToCheck = [
    # Get all of these extensions and templates (*.in)
    re.compile(r"[.](cmake|cpp|cc|cu|cuh|h|hpp|md|rst|sh|pxd|py|pyx|yml|yaml)(\.in)?$"),
    # And files with a particular file/extension combo
    re.compile(r"CMakeLists[.]txt$"),
    re.compile(r"setup[.]cfg$"),
    re.compile(r"[.]flake8[.]cython$"),
    re.compile(r"meta[.]yaml$"),
    re.compile(r"[^ \/\n]*Dockerfile[^ \/\n]*")
]

# Nothing in a build folder or .cache
ExemptFiles: typing.List[re.Pattern] = [
    r"(_version|versioneer)\.py",  # Skip versioning files
    r"^[^ \/\n]*\.cache[^ \/\n]*\/.*$",  # Ignore .cache folder
    r"^[^ \/\n]*build[^ \/\n]*\/.*$",  # Ignore any build*/ folder
    r"^external\/.*$",  # Ignore external
    r"[^ \/\n]*docs/source/(_lib|_modules|_templates)/.*$",
    r"PULL_REQUEST_TEMPLATE.md"  # Ignore the PR template
]

# this will break starting at year 10000, which is probably OK :)
CheckSimple = re.compile(r"Copyright *(?:\(c\))? *(\d{4}),? *NVIDIA C(?:ORPORATION|orporation)")
CheckDouble = re.compile(r"Copyright *(?:\(c\))? *(\d{4})-(\d{4}),? *NVIDIA C(?:ORPORATION|orporation)"  # noqa: E501
                         )
CheckApacheLic = 'Licensed under the Apache License, Version 2.0 (the "License");'


def checkThisFile(f):
    # This check covers things like symlinks which point to files that DNE
    if not (os.path.exists(f)):
        return False
    if gitutils and gitutils.isFileEmpty(f):
        return False
    for exempt in ExemptFiles:
        if exempt.search(f):
            return False
    for checker in FilesToCheck:
        if checker.search(f):
            return True
    return False


def getCopyrightYears(line):
    res = CheckSimple.search(line)
    if res:
        return (int(res.group(1)), int(res.group(1)))
    res = CheckDouble.search(line)
    if res:
        return (int(res.group(1)), int(res.group(2)))
    return (None, None)


def replaceCurrentYear(line, start, end):
    # first turn a simple regex into double (if applicable). then update years
    res = CheckSimple.sub(r"Copyright (c) \1-\1, NVIDIA CORPORATION", line)
    res = CheckDouble.sub(r"Copyright (c) {:04d}-{:04d}, NVIDIA CORPORATION".format(start, end), res)
    return res


def insertLicense(f, this_year, first_line):
    ext = os.path.splitext(f)[1].lstrip('.')

    try:
        license_text = EXT_LIC_MAPPING[ext].format(YEAR=this_year)
    except KeyError:
        return [
            f,
            0,
            "Unsupported extension {} for automatic insertion, "
            "please manually insert an Apache v2.0 header or add the file to "
            "excempted from this check add it to the 'ExemptFiles' list in "
            "the 'ci/scripts/copyright.py' file (manual fix required)".format(ext),
            None
        ]

    # If the file starts with a #! keep it as the first line
    if first_line.startswith("#!"):
        replace_line = first_line + license_text
    else:
        replace_line = "{}\n{}".format(license_text, first_line)

    return [f, 1, "License inserted", replace_line]


def checkCopyright(f,
                   update_current_year,
                   verify_apache_v2=False,
                   update_start_year=False,
                   insert_license=False,
                   git_add=False):
    """
    Checks for copyright headers and their years
    """
    errs = []
    thisYear = datetime.datetime.now().year
    lineNum = 0
    crFound = False
    apacheLicFound = not verify_apache_v2
    yearMatched = False
    with io.open(f, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
    for line in lines:
        lineNum += 1
        if not apacheLicFound:
            apacheLicFound = CheckApacheLic in line

        start, end = getCopyrightYears(line)
        if start is None:
            continue

        crFound = True
        if update_start_year:
            try:
                git_start = gitutils.determine_add_date(f).year
                if start > git_start:
                    e = [
                        f,
                        lineNum,
                        "Current year not included in the "
                        "copyright header",
                        replaceCurrentYear(line, git_start, thisYear)
                    ]
                    errs.append(e)
                    continue

            except Exception as excp:
                e = [f, lineNum, "Error determining start year from git: {}".format(excp), None]
                errs.append(e)
                continue

        if start > end:
            e = [f, lineNum, "First year after second year in the copyright header (manual fix required)", None]
            errs.append(e)
        if thisYear < start or thisYear > end:
            e = [f, lineNum, "Current year not included in the copyright header", None]
            if thisYear < start:
                e[-1] = replaceCurrentYear(line, thisYear, end)
            if thisYear > end:
                e[-1] = replaceCurrentYear(line, start, thisYear)
            errs.append(e)
        else:
            yearMatched = True
    fp.close()

    if not apacheLicFound:
        if insert_license and len(lines):
            e = insertLicense(f, thisYear, lines[0])
            crFound = True
            yearMatched = True
        else:
            e = [
                f,
                0,
                "Apache copyright header missing, if this file needs to be "
                "excempted from this check add it to the 'ExemptFiles' list in "
                "the 'ci/scripts/copyright.py' file.",
                True
            ]
        errs.append(e)

    # copyright header itself not found
    if not crFound:
        e = [f, 0, "Copyright header missing or formatted incorrectly (manual fix required)", None]
        errs.append(e)

    # even if the year matches a copyright header, make the check pass
    if yearMatched and apacheLicFound:
        errs = []

    if update_current_year or update_start_year or insert_license:
        errs_update = [x for x in errs if x[-1] is not None]
        if len(errs_update) > 0:
            logging.info("File: {}. Changing line(s) {}".format(f,
                                                                ', '.join(str(x[1]) for x in errs
                                                                          if x[-1] is not None)))
            for _, lineNum, __, replacement in errs_update:
                lines[lineNum - 1] = replacement
            with io.open(f, "w", encoding="utf-8") as out_file:
                for new_line in lines:
                    out_file.write(new_line)

            if git_add:
                gitutils.add(f)

        errs = [x for x in errs if x[-1] is None]

    return errs


def checkCopyright_main():
    """
    Checks for copyright headers in all the modified files. In case of local
    repo, this script will just look for uncommitted files and in case of CI
    it compares between branches "$PR_TARGET_BRANCH" and "current-pr-branch"
    """
    retVal = 0
    global ExemptFiles

    logging.basicConfig(level=logging.DEBUG)

    argparser = argparse.ArgumentParser("Checks for a consistent copyright header in git's modified files")
    argparser.add_argument("--update-start-year",
                           dest='update_start_year',
                           action="store_true",
                           required=False,
                           help="If set, "
                           "update the start year based on a start date parsed "
                           "on the earliest entry from `git log --follow` will "
                           "only set the year if it is less than the current "
                           "copyright year")
    argparser.add_argument("--update-current-year",
                           dest='update_current_year',
                           action="store_true",
                           required=False,
                           help="If set, "
                           "update the current year if a header is already "
                           "present and well formatted.")

    argparser.add_argument("--insert",
                           dest='insert',
                           action="store_true",
                           required=False,
                           help="If set, "
                           "inserts an Apache v2.0 license into a files "
                           "without a license, implies --verify-apache-v2")

    argparser.add_argument("--fix-all",
                           dest='fix_all',
                           action="store_true",
                           required=False,
                           help="Shortcut for setting --update-start-year --update-current-year and --insert")

    git_group = argparser.add_mutually_exclusive_group()
    git_group.add_argument("--git-modified-only",
                           dest='git_modified_only',
                           action="store_true",
                           required=False,
                           help="If set, "
                           "only files seen as modified by git will be "
                           "processed. Cannot be combined with --git-diff-commits or --git-diff-staged")
    git_group.add_argument("--git-diff-commits",
                           dest='git_diff_commits',
                           required=False,
                           nargs=2,
                           metavar='hash',
                           help="If set, "
                           "only files modified between the two given commit hashes. "
                           "Cannot be combined with --git-modified-only or --git-diff-staged")
    git_group.add_argument("--git-diff-staged",
                           dest='git_diff_staged',
                           required=False,
                           nargs="?",
                           metavar='HEAD',
                           default=None,
                           const='HEAD',
                           help="If set, "
                           "only files staged for commit. "
                           "Cannot be combined with --git-modified-only or --git-diff-commits")

    argparser.add_argument("--git-add",
                           dest='git_add',
                           action="store_true",
                           required=False,
                           help="If set, "
                           "any files auto-fixed will have `git add` run on them. ")

    argparser.add_argument("--verify-apache-v2",
                           dest='verify_apache_v2',
                           action="store_true",
                           required=False,
                           help="If set, "
                           "verifies all files contain the Apache license "
                           "in their header")
    argparser.add_argument("--exclude",
                           dest='exclude',
                           action="append",
                           required=False,
                           default=["_version\\.py"],
                           help=("Exclude the paths specified (regexp). "
                                 "Can be specified multiple times."))

    (args, dirs) = argparser.parse_known_args()
    try:
        ExemptFiles = ExemptFiles + [pathName for pathName in args.exclude]
        ExemptFiles = [re.compile(file) for file in ExemptFiles]
    except re.error as reException:
        logging.exception("Regular expression error: %s", reException, exc_info=True)
        return 1

    if args.git_modified_only:
        files = gitutils.modifiedFiles()
    elif args.git_diff_commits:
        files = gitutils.changedFilesBetweenCommits(*args.git_diff_commits)
    elif args.git_diff_staged:
        files = gitutils.stagedFiles(args.git_diff_staged)
    else:
        files = gitutils.list_files_under_source_control(ref="HEAD", *dirs)

    logging.debug("File count before filter(): %s", len(files))

    # Now filter the files down based on the exclude/include
    files = gitutils.filter_files(files, path_filter=checkThisFile)

    logging.info("Checking files (%s):\n   %s", len(files), "\n   ".join(files))

    errors = []
    for f in files:
        errors += checkCopyright(f,
                                 args.update_current_year,
                                 verify_apache_v2=(args.verify_apache_v2 or args.insert or args.fix_all),
                                 update_start_year=(args.update_start_year or args.fix_all),
                                 insert_license=(args.insert or args.fix_all),
                                 git_add=args.git_add)

    if len(errors) > 0:
        logging.info("Copyright headers incomplete in some of the files!")
        for e in errors:
            logging.error("  %s:%d Issue: %s", e[0], e[1], e[2])
        logging.info("")
        n_fixable = sum(1 for e in errors if e[-1] is not None)
        path_parts = os.path.abspath(__file__).split(os.sep)
        file_from_repo = os.sep.join(path_parts[path_parts.index("ci"):])
        if n_fixable > 0:
            logging.info(("You can run `python {} --git-modified-only "
                          "--update-current-year --insert` to fix {} of these "
                          "errors.\n").format(file_from_repo, n_fixable))
        retVal = 1
    else:
        logging.info("Copyright check passed")

    return retVal


A2_LIC_HASH = """# SPDX-FileCopyrightText: Copyright (c) {YEAR}, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""

A2_LIC_C = """/*
 * SPDX-FileCopyrightText: Copyright (c) {YEAR}, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
"""

A2_LIC_MD = """<!--
SPDX-FileCopyrightText: Copyright (c) {YEAR}, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
"""

A2_LIC_RST = """..
   SPDX-FileCopyrightText: Copyright (c) {YEAR}, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

# FilesToCheck list will allow us to assume Cmake for the txt extension
EXT_LIC_MAPPING = {
    'c': A2_LIC_C,
    'cc': A2_LIC_C,
    'cmake': A2_LIC_HASH,
    'cpp': A2_LIC_C,
    'cu': A2_LIC_C,
    'cuh': A2_LIC_C,
    'h': A2_LIC_C,
    'hpp': A2_LIC_C,
    'md': A2_LIC_MD,
    'pxd': A2_LIC_HASH,
    'py': A2_LIC_HASH,
    'pyx': A2_LIC_HASH,
    'rst': A2_LIC_RST,
    'sh': A2_LIC_HASH,
    'txt': A2_LIC_HASH,
    'yaml': A2_LIC_HASH,
    'yml': A2_LIC_HASH,
}

if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    sys.exit(checkCopyright_main())
