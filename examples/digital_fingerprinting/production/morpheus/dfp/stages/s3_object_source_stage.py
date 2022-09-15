# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import logging
import os
from datetime import date
from datetime import datetime
from datetime import timedelta

import boto3
import srf
from dateutil import tz

from morpheus.config import Config
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger("morpheus.{}".format(__name__))


def daterange(start_date: date, end_date: date):
    # Floor the start date to make it easier on day boundaries
    initial_start_date = date(start_date.year, start_date.month, start_date.day)

    for n in range(int((end_date - start_date).days)):
        yield datetime.combine(initial_start_date, datetime.min.time()) + timedelta(n), datetime.combine(
            initial_start_date, datetime.min.time()) + timedelta(n + 1)


# Putting the following functions in a common location. Should be moved
def s3_filter_duo(s3_bucket, start_date: date, end_date: date):
    filtered_objects = []
    for date_span_start, date_span_end in daterange(start_date, end_date):
        object_prefix = f'DUO_AUTH_{date_span_start:%Y}-{date_span_start:%m}-{date_span_start:%d}'

        for object in s3_bucket.objects.filter(Prefix=object_prefix):
            key_object = object.key

            # Extract the timestamp from the file name
            ts_object = key_object.split('_')[2].split('.json')[0].replace('T', ' ').replace('Z', '')
            ts_object = datetime.strptime(ts_object, '%Y-%m-%d %H:%M:%S.%f')

            if (date_span_start < ts_object < date_span_end):
                filtered_objects.append(object)

    yield filtered_objects


def s3_filter_azure(s3_bucket, start_date: date, end_date: date):
    filtered_objects = []
    for date_span_start, date_span_end in daterange(start_date, end_date):
        object_prefix = f'AZUREAD-DIAG-{date_span_start:%Y}-{date_span_start:%m}-{date_span_start:%d}'

        for object in s3_bucket.objects.filter(Prefix=object_prefix):
            key_object = object.key

            # Extract the timestamp from the file name
            ts_object = key_object.split('AZUREAD-DIAG-')[1].split('.json')[0].replace('T', ' ').replace('Z', '')
            ts_object = datetime.strptime(ts_object, '%Y-%m-%d %H.%M.%S')

            if (date_span_start < ts_object < date_span_end):
                filtered_objects.append(object)

    yield filtered_objects


#def s3_get_all(s3_bucket, prefix: str, start_date: datetime, end_date: datetime):
#    filtered_objects = []
#
#    # TODO(MDD): Should use https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.list_objects_v2
#    for obj in s3_bucket.objects.filter(Prefix=prefix):
#
#        if (obj.last_modified >= start_date and obj.last_modified < end_date):
#            filtered_objects.append(obj)
#
#    print(f"Got {len(filtered_objects)} filtered objects")
#    yield filtered_objects


def s3_object_generator(bucket_name: str, filter_func, start_date: datetime, end_date: datetime):
    # Convert the start dates to UTC
    start_date_utc = start_date.astimezone(tz.tzutc())
    end_date_utc = end_date.astimezone(tz.tzutc())

    def inner():
        aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        aws_session_token = os.environ["AWS_SESSION_TOKEN"]

        session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                aws_secret_access_key=aws_secret_access_key,
                                aws_session_token=aws_session_token)

        s3_resource_handle = session.resource('s3')
        s3_bucket = s3_resource_handle.Bucket(bucket_name)

        for object_collection in filter_func(s3_bucket, start_date_utc, end_date_utc):
            yield object_collection

        # TODO(Devin): There are too many unknowns if we only look at the object modified date; this isn't
        #   guaranteed to be consistent when juxtaposed with prefix-embedded time.
        #   As an example, suppose the objects are moved from one bucket to another, all their modified times change.
        # for object_collection in s3_get_all(s3_bucket, prefix, start_date_utc, end_date_utc):
        #    yield object_collection

    return inner


class S3BucketSourceStage(SingleOutputSource):
    """
    Source stage is used to load objects from an s3 resource and pushing them to the pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    object_generator: generator function which will produce s3 objects until exhausted
    """

    def __init__(
        self,
        c: Config,
        object_generator,
    ):
        super().__init__(c)

        self._object_generator = object_generator

    @property
    def name(self) -> str:
        return "object-from-s3"

    def supports_cpp_node(self):
        return False

    def _build_source(self, builder: srf.Builder) -> StreamPair:
        out_stream = builder.make_source(self.unique_name, self._object_generator())

        out_type = type(boto3.Session().resource('s3').Object("_", "_"))

        return out_stream, out_type
