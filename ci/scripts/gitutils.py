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
import functools
import logging
import os
import re
import subprocess
import typing


def isFileEmpty(f):
    return os.stat(f).st_size == 0


def __run_cmd(exe: str, *args: tuple[str]):
    """Runs a command with args and returns its output"""

    cmd_list = [exe] + list(args)

    # Join the args to make the command string (for logging only)
    cmd_str = " ".join(cmd_list)

    # If we only passed in one executable (could be piping commands together) then use a shell
    shell = False if len(args) > 0 else True

    if (shell):
        # For logging purposes, if we only passed in one executable, then clear the exe name to make logging better
        exe = ""

    try:
        ret = subprocess.check_output(cmd_list, stderr=subprocess.PIPE, shell=shell)
        output = ret.decode("UTF-8").rstrip("\n")

        logging.debug("Running %s command: `%s`, Output: '%s'", exe, cmd_str, output)

        return output
    except subprocess.CalledProcessError as e:
        logging.warning("Running %s command [ERRORED]: `%s`, Output: '%s'",
                        exe,
                        cmd_str,
                        e.stderr.decode("UTF-8").rstrip("\n"))
        raise


def __gh(*args):
    """Runs a Github CLI command and returns its output"""

    return __run_cmd("gh", *args)


def __git(*args):
    """Runs a git command and returns its output"""
    return __run_cmd("git", *args)


def __gitdiff(*opts):
    """Runs a git diff command with no pager set"""
    return __git("--no-pager", "diff", *opts)


def repo_version():
    """
    Determines the version of the repo by using `git describe`

    Returns
    -------
    str
        The full version of the repo in the format 'v#.#.#{a|b|rc}'
    """
    return __git("describe", "--tags", "--abbrev=0")


def add(f):
    """Runs git add on file"""
    return __git("add", f)


def determine_add_date(file_path):
    """Return the date a given file was added to git"""
    date_str = __git("log", "--follow", "--format=%as", "--", file_path, "|", "tail", "-n 1")
    return datetime.datetime.strptime(date_str, "%Y-%m-%d")


@functools.lru_cache
def gh_has_cli():
    try:
        __gh("--version")

        gh_get_repo_owner_name()

        logging.debug("Github CLI is installed")
        return True
    except subprocess.CalledProcessError:
        logging.debug("Github CLI is not installed")
        return False


@functools.lru_cache
def gh_get_repo_owner_name():

    return __gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner")


@functools.lru_cache
def gh_get_pr_number():

    return __gh("pr", "status", "--json", "number", "--jq", ".currentBranch.number")


@functools.lru_cache
def gh_get_pr_base_ref_name():

    return __gh("pr", "status", "--json", "baseRefName", "--jq", ".currentBranch.baseRefName")


@functools.lru_cache
def gh_is_pr():

    pr_number = gh_get_pr_number()

    return pr_number is not None and pr_number != ""


@functools.lru_cache
def gh_get_target_remote_branch():

    # Make sure we have the CLI
    if (not gh_has_cli()):
        return None

    # Make sure we are in a PR
    if (not gh_is_pr()):
        return None

    # Get the PR base reference
    base_ref = gh_get_pr_base_ref_name()

    # Now determine the remote ref name matching our repository
    remote_name = __run_cmd(f"git remote -v | grep :{gh_get_repo_owner_name()} | grep \"(fetch)\" | head -1 | cut -f1")

    target_ref_full_name = f"{remote_name}/{base_ref}"

    return target_ref_full_name


@functools.lru_cache
def gh_get_target_remote_branch():

    # Make sure we have the CLI
    if (not gh_has_cli()):
        return None

    # Make sure we are in a PR
    if (not gh_is_pr()):
        return None

    # Get the PR base reference
    base_ref = gh_get_pr_base_ref_name()

    # Now determine the remote ref name matching our repository
    remote_name = git_get_remote_name(gh_get_repo_owner_name())

    target_ref_full_name = f"{remote_name}/{base_ref}"

    return target_ref_full_name


@functools.lru_cache
def git_get_repo_owner_name():

    return "nv-morpheus/" + __run_cmd("git remote -v | grep -oP '/\K\w*(?=\.git \(fetch\))' | head -1")


@functools.lru_cache
def git_get_remote_name(repo_owner_and_name: str):

    return __run_cmd(f"git remote -v | grep :{repo_owner_and_name} | grep \"(fetch)\" | head -1 | cut -f1")


@functools.lru_cache
def git_get_target_remote_branch():
    try:
        # Try and guess the target branch as "branch-<major>.<minor>"
        version = git_repo_version_major_minor()

        if (version is None):
            return None

        base_ref = "branch-{}".format(version)
    except Exception:
        logging.debug("Could not determine branch version falling back to main")
        base_ref = "main"

    try:
        remote_name = git_get_remote_name(git_get_repo_owner_name())

        remote_branch = f"{remote_name}/{base_ref}"
    except Exception:

        # Try and find remote name using git
        remote_branch = __git("rev-parse", "--abbrev-ref", "--symbolic-full-name", base_ref + "@{upstream}")

    return remote_branch


@functools.lru_cache
def git_top_level_dir():
    """
    Returns the top level directory for this git repo
    """
    return __git("rev-parse", "--show-toplevel")


@functools.lru_cache
def git_current_branch():
    """Returns the name of the current branch"""
    name = __git("rev-parse", "--abbrev-ref", "HEAD")
    name = name.rstrip()
    return name


@functools.lru_cache
def git_repo_version_major_minor():
    """
    Determines the version of the repo using `git describe` and returns only
    the major and minor portion

    Returns
    -------
    str
        The partial version of the repo in the format '{major}.{minor}'
    """

    full_repo_version = repo_version()

    match = re.match(r"^v?(?P<major>[0-9]+)(?:\.(?P<minor>[0-9]+))?", full_repo_version)

    if (match is None):
        logging.debug("Could not determine repo major minor version. "
                      f"Full repo version: {full_repo_version}.")
        return None

    out_version = match.group("major")

    if (match.group("minor")):
        out_version += "." + match.group("minor")

    return out_version


def determine_merge_commit(current_branch="HEAD"):
    """
    When running outside of CI, this will estimate the target merge commit hash
    of `current_branch` by finding a common ancester with the remote branch
    'branch-{major}.{minor}' where {major} and {minor} are determined from the
    repo version.

    Parameters
    ----------
    current_branch : str, optional
        Which branch to consider as the current branch, by default "HEAD"

    Returns
    -------
    str
        The common commit hash ID
    """

    # Order of operations:
    # 1. Try to determine the target branch from GitLab CI (assuming were in a PR)
    # 2. Try to guess the target branch as "branch-<major>.<minor>" using the most recent tag (assuming we have a remote pointing to the base repo)
    # 3. Try to determine the target branch by finding a head reference that matches "branch-*" and is in this history
    # 4. Fall back to "main" if all else fails or the target branch and current branch are the same

    remote_branch = gh_get_target_remote_branch()

    if (remote_branch is None):
        # Try to use tags
        remote_branch = git_get_target_remote_branch()

    if (remote_branch is None):

        raise RuntimeError("Could not determine remote_branch. Manually set TARGET_BRANCH to continue")

    logging.debug(f"Determined TARGET_BRANCH as: '{remote_branch}'. "
                  "Finding common ancestor.")

    common_commit = __git("merge-base", remote_branch, current_branch)

    return common_commit

    try:
        repo_version = git_repo_version_major_minor()

        # Try to determine the target branch from the most recent tag
        head_branch = __git("describe", "--all", "--tags", "--match='branch-*'", "--abbrev=0")
    except subprocess.CalledProcessError:
        logging.warning("Could not determine target branch from most recent "
                        "tag. Falling back to 'branch-{major}.{minor}.")
        head_branch = None

    if (head_branch is not None):
        # Convert from head to branch name
        head_branch = __git("name-rev", "--name-only", head_branch)
    else:
        try:
            # Try and guess the target branch as "branch-<major>.<minor>"
            version = git_repo_version_major_minor()

            if (version is None):
                return None

            head_branch = "branch-{}".format(version)
        except Exception:
            logging.debug("Could not determine branch version falling back to main")
            head_branch = "main"

    try:
        # Now get the remote tracking branch
        remote_branch = __git("rev-parse", "--abbrev-ref", "--symbolic-full-name", head_branch + "@{upstream}")
    except subprocess.CalledProcessError:
        logging.debug("Could not remote tracking reference for "
                      f"branch {head_branch}.")
        remote_branch = None

    if (remote_branch is None):
        return None

    logging.debug(f"Determined TARGET_BRANCH as: '{remote_branch}'. "
                  "Finding common ancestor.")

    common_commit = __git("merge-base", remote_branch, current_branch)

    return common_commit


def uncommittedFiles():
    """
    Returns a list of all changed files that are not yet committed. This
    means both untracked/unstaged as well as uncommitted files too.
    """
    files = __git("status", "-u", "-s")
    ret = []
    for f in files.splitlines():
        f = f.strip(" ")
        f = re.sub("\s+", " ", f)  # noqa: W605
        tmp = f.split(" ", 1)
        # only consider staged files or uncommitted files
        # in other words, ignore untracked files
        if tmp[0] == "M" or tmp[0] == "A":
            ret.append(tmp[1])
    return ret


def changedFilesBetween(baseName, branchName, commitHash):
    """
    Returns a list of files changed between branches baseName and latest commit
    of branchName.
    """
    current = git_current_branch()
    # checkout "base" branch
    __git("checkout", "--force", baseName)
    # checkout branch for comparing
    __git("checkout", "--force", branchName)
    # checkout latest commit from branch
    __git("checkout", "-fq", commitHash)

    files = __gitdiff("--name-only", "--ignore-submodules", f"{baseName}..{branchName}")

    # restore the original branch
    __git("checkout", "--force", current)
    return files.splitlines()


def is_repo_relative(f: str, git_root: str = None):
    if (git_root is None):
        git_root = git_top_level_dir()

    abs_f = os.path.abspath(f)

    rel_path = os.path.relpath(abs_f, git_root)

    return not rel_path.startswith("../")


def filter_files(files: typing.Union[str, typing.List[str]], path_filter=None):
    # Convert all to array of strings
    if (isinstance(files, str)):
        files = files.splitlines()

    git_root = git_top_level_dir()

    ret_files = []
    for fn in files:
        # Check that we are relative to the git repo
        assert is_repo_relative(fn, git_root=git_root), f"Path {fn} must be relative to git root: {git_root}"

        if (path_filter is None or path_filter(fn)):
            ret_files.append(fn)

    return ret_files


def changesInFileBetween(file, b1, b2, pathFilter=None):
    """Filters the changed lines to a file between the branches b1 and b2"""
    current = git_current_branch()
    __git("checkout", "--quiet", b1)
    __git("checkout", "--quiet", b2)
    diffs = __gitdiff("--ignore-submodules", "-w", "--minimal", "-U0", "%s...%s" % (b1, b2), "--", file)
    __git("checkout", "--quiet", current)
    return filter_files(diffs, pathFilter)


def modifiedFiles(pathFilter=None):
    """
    If inside a CI-env (ie. TARGET_BRANCH and COMMIT_HASH are defined, and
    current branch is "current-pr-branch"), then lists out all files modified
    between these 2 branches. Locally, TARGET_BRANCH will try to be determined
    from the current repo version and finding a coresponding branch named
    'branch-{major}.{minor}'. If this fails, this functino will list out all
    the uncommitted files in the current branch.

    Such utility function is helpful while putting checker scripts as part of
    cmake, as well as CI process. This way, during development, only the files
    touched (but not yet committed) by devs can be checked. But, during the CI
    process ALL files modified by the dev, as submiited in the PR, will be
    checked. This happens, all the while using the same script.
    """
    targetBranch = os.environ.get("TARGET_BRANCH")
    commitHash = os.environ.get("COMMIT_HASH")
    currentBranch = git_current_branch()
    logging.info("TARGET_BRANCH=%s, COMMIT_HASH=%s, currentBranch=%s", targetBranch, commitHash, currentBranch)

    if targetBranch and commitHash and (currentBranch == "current-pr-branch"):
        logging.debug("Assuming a CI environment.")
        allFiles = changedFilesBetween(targetBranch, currentBranch, commitHash)
    else:
        logging.debug("Did not detect CI environment. "
                      "Determining TARGET_BRANCH locally.")

        common_commit = determine_merge_commit(currentBranch)

        if (common_commit is not None):

            # Now get the diff. Use --staged to get both diff between
            # common_commit..HEAD and any locally staged files
            allFiles = __gitdiff("--name-only", "--ignore-submodules", "--staged", f"{common_commit}").splitlines()
        else:
            # Fallback to just uncommitted files
            allFiles = uncommittedFiles()

    files = []
    for f in allFiles:
        if pathFilter is None or pathFilter(f):
            files.append(f)

    filesToCheckString = "\n\t".join(files) if files else "<None>"
    logging.debug(f"Found files to check:\n\t{filesToCheckString}\n")
    return files


def changedFilesBetweenCommits(base_commit, commit, pathFilter=None):
    diffs = __gitdiff("--name-only", f"{base_commit}...{commit}")
    return filter_files(diffs, pathFilter)


def stagedFiles(base='HEAD', pathFilter=None):
    diffs = __gitdiff("--cached", "--name-only", base)
    return filter_files(diffs, pathFilter)


def list_files_under_source_control(*paths: str, ref: str = None):

    # Use HEAD if no ref is supplied
    if (ref is None):
        ref = "HEAD"

    git_args = ["ls-tree", "-r", "--name-only", ref] + list(paths)

    return __git(*git_args).split("\n")


def listAllFilesInDir(folder):
    """Utility function to list all files/subdirs in the input folder"""
    allFiles = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            allFiles.append(os.path.join(root, name))
    return allFiles


def listFilesToCheck(filesDirs, filter=None):
    """
    Utility function to filter the input list of files/dirs based on the input
    filter method and returns all the files that need to be checked
    """
    allFiles = []
    for f in filesDirs:
        if os.path.isfile(f):
            if filter is None or filter(f):
                allFiles.append(f)
        elif os.path.isdir(f):
            files = listAllFilesInDir(f)
            for f_ in files:
                if filter is None or filter(f_):
                    allFiles.append(f_)
    return allFiles


def get_merge_target():
    currentBranch = git_current_branch()
    return determine_merge_commit(currentBranch)


def parse_args():
    argparser = argparse.ArgumentParser("Executes a gitutil action")
    argparser.add_argument("action", choices=['get_merge_target'], help="Action to execute")
    args = argparser.parse_args()
    return args


def main():
    log_level = logging.getLevelName(os.environ.get("MORPHEUS_LOG_LEVEL", "WARNING"))
    logging.basicConfig(format="%(levelname)s:%(message)s", level=log_level)

    args = parse_args()

    if args.action == 'get_merge_target':
        print(get_merge_target())


if __name__ == '__main__':
    main()
