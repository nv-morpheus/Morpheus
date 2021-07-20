# Copyright (c) 2021, NVIDIA CORPORATION.
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
import cudf
import numba as nb
from cuml import ForestInference

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        preds_config = pb_utils.get_output_config_by_name(model_config, "preds")

        self.preds_config = pb_utils.triton_string_to_numpy(preds_config['data_type'])

        model_filepath = '/models/anomaly_detection_fil_model/1/xgboost.model'

        self.column_names_dct = {
            0: 'ack',
            1: 'psh',
            2: 'rst',
            3: 'syn',
            4: 'fin',
            5: 'ppm',
            6: 'data_len',
            7: 'bpp',
            8: 'all',
            9: 'ackpush/all',
            10: 'rst/all',
            11: 'syn/all',
            12: 'fin/all'
        }

        self.model = ForestInference.load(model_filepath, output_class=True)

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get input data
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input__0")
            input_np = input_tensor.as_numpy()
            src = nb.cuda.to_device(input_np)
            df = cudf.DataFrame(src)
            df = df.rename(columns=self.column_names_dct)
            # predict model
            pred_series = self.model.predict(df)
            preds_np = pred_series.to_array()
            preds_tensor = pb_utils.Tensor("preds", preds_np.astype(self.preds_config))

            inference_response = pb_utils.InferenceResponse(output_tensors=[preds_tensor])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
