import re
from datetime import datetime
from datetime import timezone

import fsspec

iso_date_regex = re.compile(r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
                            r"T(?P<hour>\d{1,2}):(?P<minute>\d{1,2}):(?P<second>\d{1,2})(?P<microsecond>\.\d{1,6})?Z")


def date_extractor(file_object: fsspec.core.OpenFile, filename_regex: re.Pattern):

    assert isinstance(file_object, fsspec.core.OpenFile)

    file_path = file_object.path

    # Match regex with the pathname since that can be more accurate
    match = filename_regex.search(file_path)

    if (match):
        # Convert the regex match
        groups = match.groupdict()

        if ("microsecond" in groups):
            groups["microsecond"] = int(float(groups["microsecond"]) * 1000000)

        groups = {key: int(value) for key, value in groups.items()}

        groups["tzinfo"] = timezone.utc

        ts_object = datetime(**groups)
    else:
        # Otherwise, fallback to the file modified (created?) time
        ts_object = file_object.fs.modified(file_object.path)

    return ts_object
