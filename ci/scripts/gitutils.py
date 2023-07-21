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
import json
import logging
import os
import re
import subprocess
import typing


def _run_cmd(exe: str, *args: tuple[str]):
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


def _gh(*args):
    """Runs a Github CLI command and returns its output"""

    return _run_cmd("gh", *args)


def _git(*args):
    """Runs a git command and returns its output"""
    return _run_cmd("git", *args)


class GitUtils:

    @functools.lru_cache
    @staticmethod
    def get_closest_tag():
        """
        Determines the version of the repo by using `git describe`

        Returns
        -------
        str
            The full version of the repo in the format 'v#.#.#{a|b|rc}'
        """
        return _git("describe", "--tags", "--abbrev=0")

    @functools.lru_cache
    @staticmethod
    def get_repo_version():
        """
        Determines the version of the repo using `git describe` and returns only
        the major and minor portion

        Returns
        -------
        str
            The partial version of the repo in the format '{major}.{minor}'
        """

        full_repo_version = GitUtils.get_closest_tag()

        match = re.match(r"^v?(?P<major>[0-9]+)(?:\.(?P<minor>[0-9]+))?", full_repo_version)

        if (match is None):
            logging.debug("Could not determine repo major minor version. Full repo version: %s.", full_repo_version)
            return None

        out_version = match.group("major")

        if (match.group("minor")):
            out_version += "." + match.group("minor")

        return out_version

    @functools.lru_cache
    @staticmethod
    def get_repo_owner_name():

        # pylint: disable=anomalous-backslash-in-string
        return "nv-morpheus/" + _run_cmd("git remote -v | grep -oP '/\K\w*(?=\.git \(fetch\))' | head -1")  # noqa: W605

    @functools.lru_cache
    @staticmethod
    def get_repo_remote_name(repo_owner_and_name: str):

        return _run_cmd(f"git remote -v | grep :{repo_owner_and_name} | grep \"(fetch)\" | head -1 | cut -f1")

    @functools.lru_cache
    @staticmethod
    def is_ref_valid(git_ref: str):

        try:
            return _git("rev-parse", "--verify", git_ref) != ""
        except subprocess.CalledProcessError:
            return False

    @functools.lru_cache
    @staticmethod
    def get_remote_branch(local_branch_ref: str, *, repo_owner_and_name: str = None):

        if (repo_owner_and_name is None):
            repo_owner_and_name = GitUtils.get_repo_owner_name()

        remote_name = GitUtils.get_repo_remote_name(repo_owner_and_name)

        remote_branch_ref = f"{remote_name}/{local_branch_ref}"

        if (GitUtils.is_ref_valid(remote_branch_ref)):
            return remote_branch_ref

        logging.info("Remote branch '%s' for repo '%s' does not exist. Falling back to rev-parse",
                     remote_branch_ref,
                     repo_owner_and_name)

        remote_branch_ref = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", local_branch_ref + "@{upstream}")

        return remote_branch_ref

    @functools.lru_cache
    @staticmethod
    def get_target_remote_branch():
        try:
            # Try and guess the target branch as "branch-<major>.<minor>"
            version = GitUtils.get_repo_version()

            if (version is None):
                return None

            base_ref = f"branch-{version}"

            # If our current branch and the base ref are the same, then use main
            if (base_ref == GitUtils.git_current_branch()):
                logging.warning("Current branch is the same as the tagged branch: %s. Falling back to 'main'", base_ref)
                base_ref = "main"

        except Exception:
            logging.debug("Could not determine branch version falling back to main")
            base_ref = "main"

        return GitUtils.get_remote_branch(base_ref)

    @functools.lru_cache
    @staticmethod
    def get_repo_dir():
        """
        Returns the top level directory for this git repo
        """
        return _git("rev-parse", "--show-toplevel")

    @functools.lru_cache
    @staticmethod
    def get_current_branch():
        """Returns the name of the current branch"""
        name = _git("rev-parse", "--abbrev-ref", "HEAD")
        name = name.rstrip()
        return name

    @staticmethod
    def add_files(*files_to_add):
        """Runs git add on file"""
        return _git("add", *files_to_add)

    @functools.lru_cache
    @staticmethod
    def get_file_add_date(file_path):
        """Return the date a given file was added to git"""
        date_str = _git("log", "--follow", "--format=%as", "--", file_path, "|", "tail", "-n 1")
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")

    @staticmethod
    def get_uncommitted_files():
        """
        Returns a list of all changed files that are not yet committed. This
        means both untracked/unstaged as well as uncommitted files too.
        """
        files = _git("status", "-u", "-s")
        ret = []
        for f in files.splitlines():
            f = f.strip(" ")
            f = re.sub(r"\s+", " ", f)  # noqa: W605
            tmp = f.split(" ", 1)
            # only consider staged files or uncommitted files
            # in other words, ignore untracked files
            if tmp[0] == "M" or tmp[0] == "A":
                ret.append(tmp[1])
        return ret

    @staticmethod
    def diff(base_ref: str, target_ref: str, merge_base: bool = False, staged: bool = False):

        assert base_ref is not None or base_ref != "", "base_ref must be a valid ref"
        assert target_ref is not None or target_ref != "", "target_ref must be a valid ref"

        args = ["--no-pager", "diff", "--name-only", "--ignore-submodules"]

        if (merge_base):
            args.append("--merge-base")

        if (staged):
            args.append("--cached")

        args += [base_ref, target_ref]

        return _git(*args).splitlines()

    @staticmethod
    def diff_index(target_ref: str, merge_base: bool = False, staged: bool = False):

        assert target_ref is not None or target_ref != "", "target_ref must be a valid ref"

        args = ["--no-pager", "diff-index", "--name-only", "--ignore-submodules"]

        if (merge_base):
            args.append("--merge-base")

        if (staged):
            args.append("--cached")

        args += [target_ref]

        return _git(*args).splitlines()


class GithubUtils:

    @functools.lru_cache
    @staticmethod
    def has_cli():
        try:
            _gh("--version")

            # Run a test function
            repo_name = _gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner")

            logging.debug("Github CLI is installed. Using repo: %s", repo_name)
            return True
        except subprocess.CalledProcessError:
            logging.debug("Github CLI is not installed")
            return False

    @functools.lru_cache
    @staticmethod
    def get_repo_owner_name():

        # Make sure we have the CLI
        if (not GithubUtils.has_cli()):
            return None

        return _gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner")

    @functools.lru_cache
    @staticmethod
    def get_pr_info() -> dict:

        # Make sure we have the CLI
        if (not GithubUtils.has_cli()):
            return None

        # List of fields to get from the PR
        fields = [
            "baseRefName",
            "number",
        ]

        json_output = _gh("pr", "status", "--json", ",".join(fields), "--jq", ".currentBranch")

        if (json_output == ""):
            return None

        return json.loads(json_output)

    @functools.lru_cache
    @staticmethod
    def is_pr():

        return GithubUtils.get_pr_info() is not None

    @functools.lru_cache
    @staticmethod
    def get_pr_number():

        pr_info = GithubUtils.get_pr_info()

        if (pr_info is None):
            return None

        return pr_info["number"]

    @functools.lru_cache
    @staticmethod
    def get_pr_base_ref_name():

        pr_info = GithubUtils.get_pr_info()

        if (pr_info is None):
            return None

        return pr_info["baseRefName"]

    @functools.lru_cache
    @staticmethod
    def get_pr_target_remote_branch():

        # Make sure we are in a PR
        if (not GithubUtils.is_pr()):
            return None

        # Get the PR base reference
        base_ref = GithubUtils.get_pr_base_ref_name()

        # Now determine the remote ref name matching our repository
        remote_name = GitUtils.get_remote_branch(base_ref, repo_owner_and_name=GithubUtils.get_repo_owner_name())

        return remote_name


def _is_repo_relative(f: str, git_root: str = None):
    if (git_root is None):
        git_root = GitUtils.get_repo_dir()

    abs_f = os.path.abspath(f)

    rel_path = os.path.relpath(abs_f, git_root)

    return not rel_path.startswith("../")


# def modified_files(path_filter=None):
#     """
#     If inside a CI-env (ie. TARGET_BRANCH and COMMIT_HASH are defined, and
#     current branch is "current-pr-branch"), then lists out all files modified
#     between these 2 branches. Locally, TARGET_BRANCH will try to be determined
#     from the current repo version and finding a coresponding branch named
#     'branch-{major}.{minor}'. If this fails, this functino will list out all
#     the uncommitted files in the current branch.

#     Such utility function is helpful while putting checker scripts as part of
#     cmake, as well as CI process. This way, during development, only the files
#     touched (but not yet committed) by devs can be checked. But, during the CI
#     process ALL files modified by the dev, as submiited in the PR, will be
#     checked. This happens, all the while using the same script.
#     """
#     target_branch = os.environ.get("TARGET_BRANCH")
#     commit_hash = os.environ.get("COMMIT_HASH")
#     current_branch = GitUtils.get_current_branch()
#     logging.info("TARGET_BRANCH=%s, COMMIT_HASH=%s, currentBranch=%s", target_branch, commit_hash, current_branch)

#     if target_branch and commit_hash and (current_branch == "current-pr-branch"):
#         logging.debug("Assuming a CI environment.")
#         all_files = _changed_files_between(target_branch, current_branch, commit_hash)
#     else:
#         logging.debug("Did not detect CI environment. "
#                       "Determining TARGET_BRANCH locally.")

#         common_commit = determine_merge_commit(current_branch)

#         if (common_commit is not None):

#             # Now get the diff. Use --staged to get both diff between
#             # common_commit..HEAD and any locally staged files
#             all_files = _gitdiff("--name-only", "--ignore-submodules", "--staged", f"{common_commit}").splitlines()
#         else:
#             # Fallback to just uncommitted files
#             all_files = GitUtils.get_uncommitted_files()

#     files = []
#     for f in all_files:
#         if path_filter is None or path_filter(f):
#             files.append(f)

#     files_to_check_string = "\n\t".join(files) if files else "<None>"

#     logging.debug("Found files to check:\n\t%s\n", files_to_check_string)

#     return files


def get_merge_target():
    # Returns the merge target branch for the current branch as if it were a PR/MR

    # Order of operations:
    # 1. Try to determine the target branch from GitLab CI (assuming were in a PR)
    # 2. Try to guess the target branch as "branch-<major>.<minor>" using the most recent tag (assuming we have a remote
    #    pointing to the base repo)
    # 3. Try to determine the target branch by finding a head reference that matches "branch-*" and is in this history
    # 4. Fall back to "main" if all else fails or the target branch and current branch are the same
    remote_branch = GithubUtils.get_pr_target_remote_branch()

    if (remote_branch is None):
        # Try to use tags
        remote_branch = GitUtils.get_target_remote_branch()

    if (remote_branch is None):

        raise RuntimeError("Could not determine remote_branch. Manually set TARGET_BRANCH to continue")

    return remote_branch


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

    remote_branch = get_merge_target()

    common_commit = _git("merge-base", remote_branch, current_branch)

    logging.info("Determined TARGET_BRANCH as: '%s'. With merge-commit: %s", remote_branch, common_commit)

    return common_commit


def filter_files(files: typing.Union[str, typing.List[str]], path_filter=None):
    # Convert all to array of strings
    if (isinstance(files, str)):
        files = files.splitlines()

    git_root = GitUtils.get_repo_dir()

    ret_files = []
    for file in files:
        # Check that we are relative to the git repo
        assert _is_repo_relative(file, git_root=git_root), f"Path {file} must be relative to git root: {git_root}"

        if (path_filter is None or path_filter(file)):
            ret_files.append(file)

    return ret_files


def changed_files(target_ref=None, base_ref="HEAD", *, merge_base: bool = True, staged=False, path_filter=None):
    # Comparison between 2 commits (ignoring the working tree)

    if (target_ref is None):
        target_ref = os.environ.get("TARGET_BRANCH", None)

    if (target_ref is None):
        target_ref = get_merge_target()

    logging.info("Comparing %s..%s with merge_base: %s, staged: %s", base_ref, target_ref, merge_base, staged)

    diffs = GitUtils.diff(base_ref, target_ref, merge_base=merge_base, staged=staged)

    return filter_files(diffs, path_filter=path_filter)


def modified_files(target_ref=None, *, merge_base: bool = True, staged=False, path_filter=None):
    # Comparison between the working tree (files on system) and a commit

    if (target_ref is None):
        target_ref = os.environ.get("TARGET_BRANCH", None)

    if (target_ref is None):
        target_ref = get_merge_target()

    logging.info("Comparing index to %s with merge_base: %s, staged: %s", target_ref, merge_base, staged)

    diffs = GitUtils.diff_index(target_ref, merge_base=merge_base, staged=staged)

    return filter_files(diffs, path_filter=path_filter)


def staged_files(base_ref="HEAD", *, path_filter=None):

    return modified_files(target_ref=base_ref, merge_base=False, staged=True, path_filter=path_filter)


def all_files(*paths, base_ref="HEAD", path_filter=None):

    git_args = ["ls-tree", "-r", "--name-only", base_ref] + list(paths)

    ls_files = _git(*git_args)

    return filter_files(ls_files, path_filter=path_filter)


def add_files(*files_to_add):
    return GitUtils.add_file(*files_to_add)


def get_file_add_date(filename: str):

    return GitUtils.get_file_add_date(filename)


def _parse_args():
    argparser = argparse.ArgumentParser("Executes a gitutil action")
    argparser.add_argument("action", choices=['get_merge_target'], help="Action to execute")
    args = argparser.parse_args()
    return args


def _main():
    log_level = logging.getLevelName(os.environ.get("MORPHEUS_LOG_LEVEL", "WARNING"))
    logging.basicConfig(format="%(levelname)s:%(message)s", level=log_level)

    args = _parse_args()

    if args.action == 'get_merge_target':
        print(determine_merge_commit())


if __name__ == '__main__':
    _main()
