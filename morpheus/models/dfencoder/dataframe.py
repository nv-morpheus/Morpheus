import pandas as pd
import numpy as np

class EncoderDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(EncoderDataFrame, self).__init__(*args, **kwargs)

    def swap(self, likelihood=.15):
        """
        Performs random swapping of data.
        Each value has a likelihood of *argument likelihood*
            of being randomly replaced with a value from a different
            row.
        Returns a copy of the dataframe with equal size.
        """

        #select values to swap
        tot_rows = self.__len__()
        n_rows = int(round(tot_rows*likelihood))
        n_cols = len(self.columns)

        def gen_indices():
            column = np.repeat(np.arange(n_cols).reshape(1, -1), repeats=n_rows, axis=0)
            row = np.random.randint(0, tot_rows, size=(n_rows, n_cols))
            return row, column

        row, column = gen_indices()
        new_mat = self.values
        to_place = new_mat[row, column]

        row, column = gen_indices()
        new_mat[row, column] = to_place

        dtypes = {col:typ for col, typ in zip(self.columns, self.dtypes)}
        result = EncoderDataFrame(columns=self.columns, data=new_mat)
        result = result.astype(dtypes, copy=False)

        return result
