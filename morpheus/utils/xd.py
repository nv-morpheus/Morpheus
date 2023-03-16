import typing

import pandas as pd

import cudf

Series = typing.Union[cudf.Series, pd.Series]
DataFrame = typing.Union[cudf.DataFrame, pd.DataFrame]
