import os

# Set RAPIDS_NO_INITIALIZE to prevent cuda from being initialized
# See issue https://github.com/rapidsai/cudf/issues/4827
os.environ["RAPIDS_NO_INITIALIZE"] = "1"

# Now run the imports
import cudf
import cuml

import morpheus
