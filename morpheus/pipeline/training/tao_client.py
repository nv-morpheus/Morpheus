# Copyright (c) 2023, NVIDIA CORPORATION.
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

import json
import logging
import os
import typing

import requests

logger = logging.getLogger("morpheus.{}".format(__name__))

_KIND = typing.Literal["model", "dataset"]
_ACTIONS = typing.Literal["convert", "train", "evaluate", "prune", "retrain", "export", "inference"]


def generate_schema_url(url, ssl):
    if url.startswith("http://") or url.startswith("https://"):
        raise ValueError("URL should not include the scheme")

    scheme = "https://" if ssl else "http://"
    url = scheme + (url if url[-1] != "/" else url[:-1])

    return url


def vaildate_apikey(apikey):
    if not isinstance(apikey, str):
        raise ValueError('API key must be a string')

    if not apikey:
        raise ValueError('API key can not be an empty string')

    return apikey


class TaoApiClient():

    def __init__(self,
                 apikey: str,
                 url: str,
                 ssl: bool = False,
                 cert: str = None,
                 server_side_cert: bool = True,
                 proxies: typing.Dict[str, str] = None):

        self._apikey = vaildate_apikey(apikey)
        self._parsed_url = generate_schema_url(url, ssl)
        base_uri = self._parsed_url.rstrip('/')
        self._base_uri = f"{base_uri}/api/v1"

        self._session = requests.Session()

        login_creds = self._login()

        self._user_uri = self._base_uri + "/user/" + login_creds.get("user_id")

        if not ssl:
            self._session.headers.update({'Authorization': 'Bearer ' + login_creds.get("token")})

        else:
            if server_side_cert:
                self._session.verify = cert
            self._session.cert = cert

        if proxies:
            self._session.proxies.update(proxies)

    def _login(self):

        endpoint = f"{self._base_uri}/login/{self._apikey}"

        logger.debug("Login endpoint: {}".format(endpoint))

        resp = self._session.get(endpoint)
        if not resp.status_code == 200:
            raise Exception("Login failed: {}".format(resp.reason))

        logger.info("Login has been successful!")

        return json.loads(resp.content)

    @property
    def base_uri(self):
        return self._base_uri

    @property
    def user_uri(self):
        return self._user_uri

    @property
    def session(self):
        return self._session

    def create_resource(self, data: typing.Dict, kind: _KIND, **kwargs) -> str:
        """
        Create new resource.

        Parameters
        ----------
        data : typing.Dict
            Initial metadata for new resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        resource_id: str
            Unique identifier for the resource.
        """

        data = json.dumps(data)

        endpoint = f"{self.user_uri}/{kind}"

        logger.debug("reate resource with endpoint: {}".format(endpoint))

        resp = self.session.post(endpoint, data=data, **kwargs)
        if not resp.status_code == 201:
            raise Exception("Error creating resource {} with endpoint {}: {}".format(kind, endpoint, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        resource_id = json_resp.get("id")

        return resource_id

    def partial_update_resource(self, data: typing.Dict, resource_id: str, kind: _KIND, **kwargs) -> typing.Dict:
        """
        Partially update the resource.

        Parameters
        ----------
        data : typing.Dict
            Metadata that needs to be updated.
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        data = json.dumps(data)

        endpoint = f"{self.user_uri}/{kind}/{resource_id}"
        logger.debug("Partially update resource with endpoint: {}".format(endpoint))

        resp = self.session.patch(endpoint, data=data, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Unable to partially update resource: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def update_resource(self, data: typing.Dict, resource_id: str, kind: _KIND, **kwargs) -> typing.Dict:
        """
        Update the resource.

        Parameters
        ----------
        data : typing.Dict
            Metadata that needs to be updated.
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        data = json.dumps(data)

        endpoint = f"{self.user_uri}/{kind}/{resource_id}"
        logger.debug("Update resource with endpoint: {}".format(endpoint))

        resp = self.session.put(endpoint, data=data, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Unable to update resource: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def upload_resource(self, resource_path: str, resource_id: str, kind: _KIND, **kwargs) -> typing.Dict:
        """
        Upload the resource.

        Parameters
        ----------
        resource_path : str
            The location of the resource to be uploaded.
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        if os.path.exists(resource_path):
            if os.path.isfile(resource_path):
                files = [("file", open(resource_path, "rb"))]
            else:
                raise Exception("Resource path must be a file.")
        else:
            raise ValueError("Resource path provided does not exists.")

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/upload"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.post(endpoint, files=files, **kwargs)
        if not resp.status_code == 201:
            raise Exception("Unable to upload resource: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def list_resources(self, kind: _KIND, **kwargs) -> typing.Dict:
        """
        List available resources by kind.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Unable to list the resources: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def get_specs_schema(self, resource_id: str, kind: _KIND, action: _ACTIONS, **kwargs) -> typing.Dict:
        """
        Get specs schema by kind and action.

        Parameters
        ----------
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        action: _ACTIONS
            TAO actions.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/specs/{action}/schema"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Error getting specs schema: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def get_specs(self, resource_id: str, kind: _KIND, action: _ACTIONS, **kwargs) -> typing.Dict:
        """
        Get specs by kind and action.

        Parameters
        ----------
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        action: _ACTIONS
            TAO actions.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/specs/{action}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Error getting specs: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def update_specs(self, specs: typing.Dict, resource_id: str, kind: _KIND, action: _ACTIONS,
                     **kwargs) -> typing.Dict:
        """
        Update specs by kind and action.

        Parameters
        ----------
        specs: typing.Dict
            Updated specs.
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        action: _ACTIONS
            TAO actions.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/specs/{action}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        data = json.dumps(specs)

        resp = self.session.post(endpoint, data=data, **kwargs)
        if not resp.status_code == 201:
            raise Exception("Unable to update specs: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def save_specs(self, resource_id: str, kind: _KIND, action: _ACTIONS, **kwargs) -> typing.Dict:
        """
        Save specs by kind and action.

        Parameters
        ----------
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        action: _ACTIONS
            TAO actions.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/specs/{action}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.post(endpoint, **kwargs)
        if not resp.status_code == 201:
            raise Exception("Error saving specs: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def run_job(self,
                resource_id: str,
                kind: _KIND,
                actions: typing.List[str],
                parent_job_id: str = None,
                **kwargs) -> typing.Dict:
        """
        Run job by kind and action.

        Parameters
        ----------
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        action: _ACTIONS
            TAO actions.
        parent_job_id: str
            Parent job id.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        data = json.dumps({"job": parent_job_id, "actions": actions})

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/job"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.post(endpoint, data=data, **kwargs)
        if not resp.status_code == 201:
            raise Exception("Unable to run the job: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def get_job_status(self, job_id: str, resource_id: str, kind: _KIND, **kwargs) -> typing.Dict:
        """
        Get job status.

        Parameters
        ----------
        job_id: str
            Unique identifier for the job.
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/job/{job_id}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Unable to retrieve job status: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def list_jobs(self, resource_id: str, kind: _KIND, **kwargs) -> typing.Dict:
        """
        List jobs for a given resource by kind.

        Parameters
        ----------
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/job"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)

        if not resp.status_code == 200:
            raise Exception("Error retrieving list of jobs belongs to {}: {}".format(kind, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def delete_job(self, resource_id: str, job_id: str, kind: _KIND, **kwargs) -> typing.Dict:
        """
        Delete job for a given kind and resource identifier.

        Parameters
        ----------
        resource_id: str
            Unique identifier for the resource.
        job_id: str
            Unique identifier for the job.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/job/{job_id}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.delete(endpoint, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Unable to delete job belongs to {} group: {}".format(kind, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def cancel_job(self, resource_id: str, job_id: str, kind: _KIND, **kwargs) -> typing.Dict:
        """
        Cancel job for a given kind and resource identifier.

        Parameters
        ----------
        resource_id: str
            Unique identifier for the resource.
        job_id: str
            Unique identifier for the job.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/job/{job_id}/cancel"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.post(endpoint, **kwargs)

        if not resp.status_code == 200:
            raise Exception("Unable to cancel {} job: {}".format(kind, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def resume_model_job(self, model_id: str, job_id: str, **kwargs) -> typing.Dict:
        """
        Resume model job.

        Parameters
        ----------
        model_id: str
            Unique identifier for the model.
        job_id: str
            Unique identifier for the job.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/model/{model_id}/job/{job_id}/resume"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.post(endpoint, **kwargs)

        if not resp.status_code == 200:
            raise Exception("Error resuming model job: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def download_resource(self, resource_id, job_id, kind: _KIND, output_dir: str, **kwargs) -> str:
        """
        Download resources.

        Parameters
        ----------
        resource_id: str
            Unique identifier for the resource.
        job_id: str
            Unique identifier for the job.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        output_dir : str
            Output directory to save the downloaded content.
        **kwargs :
            Additional arguments.
        Returns
        -------
        downloaded_path : str
            The download location's path.
        """
        job_status = self.get_job_status(resource_id=resource_id, job_id=job_id, kind=kind)

        status = job_status.get("status")

        if status == "Done":

            endpoint = f'{self.user_uri}/{kind}/{resource_id}/job/{job_id}/download'
            logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

            resp = self.session.get(endpoint, **kwargs)

            if not resp.status_code == 200:
                raise Exception("Error downloading the job content: {}".format(resp.content))

            temptar = f'{job_id}.tar.gz'

            with open(temptar, 'wb') as f:
                f.write(resp.content)
            logger.debug("Untarring {}...".format(temptar))
            tar_command = f"tar -xvf {temptar} -C {output_dir}/"
            os.system(tar_command)
            logger.debug("Untarring {}... Done".format(temptar))
            os.remove(temptar)
            downloaded_path = f"{output_dir}/{job_id}"

            logger.debug("Results at location {}".format(downloaded_path))

            return downloaded_path

        logger.info("Resource can be downloaded only when the job is completed. Current status is in {}".format(status))

    def delete_resource(self, resource_id: str, kind: _KIND, **kwargs) -> typing.Dict:
        """
        Delete resource.

        Parameters
        ----------
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """
        endpoint = f"{self.user_uri}/{kind}/{resource_id}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.delete(endpoint, **kwargs)

        if not resp.status_code == 200:
            raise Exception("Error deleting resource from {}: {}".format(kind, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def retrieve_resource(self, resource_id: str, kind: _KIND, **kwargs) -> typing.Dict:
        """
        Retrieve resource metadata.

        Parameters
        ----------
        resource_id: str
            Unique identifier for the resource.
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """
        endpoint = f"{self.user_uri}/{kind}/{resource_id}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)

        if not resp.status_code == 200:
            raise Exception("Error retrieving resource from {}: {}".format(kind, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def close(self):
        """
        Closes session.
        """
        session = getattr(self, '_session', None)
        if session:
            logger.debug("Closing session...")
            session.close()
            self._session = None
            logger.debug("Closing session... Done")
