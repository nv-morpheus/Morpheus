import cudf
import io
import os
import pandas as pd
import pytest
import unittest
import uuid

from morpheus.io import data_manager

DataManager = data_manager.DataManager


# Work around because classes that inherit unittest.TestCase don't work with parametrize
def parameterized_expand(arg_name, arg_values):
    def decorator(test_method):
        def wrapper(self, *args, **kwargs):
            for arg_value in arg_values:
                kwargs_copy = kwargs.copy()
                kwargs_copy[arg_name] = arg_value
                print(f"Running test with {args}={kwargs_copy}")
                result = test_method(self, *args, **kwargs_copy)
            return result

        return wrapper

    return decorator


sources = [
    cudf.DataFrame({'a': [1, 2], 'b': [3, 4]}),
    'buffer2.parquet',  # Local or remote file path
    pd.DataFrame({'a': [5, 6], 'b': [7, 8]}),
]
test_cudf_dataframe = cudf.DataFrame({'a': [9, 10], 'b': [11, 12]})
test_pd_dataframe = pd.DataFrame({'a': [13, 14], 'b': [15, 16]})
test_parquet_filepath = 'test_file.parquet'
test_csv_filepath = 'test_file.csv'


def setUpModule():
    global sources, test_cudf_dataframe, test_pd_dataframe, test_parquet_filepath, test_csv_filepath

    test_cudf_dataframe.to_parquet(test_parquet_filepath)
    test_cudf_dataframe.to_csv(test_csv_filepath, index=False, header=True)


def tearDownModule():
    if os.path.exists(test_parquet_filepath):
        os.remove(test_parquet_filepath)
    if os.path.exists(test_csv_filepath):
        os.remove(test_csv_filepath)

    # def setUp(self):
    #    self.sources = [
    #        cudf.DataFrame({'a': [1, 2], 'b': [3, 4]}),
    #        'buffer2.parquet',  # Local or remote file path
    #        pd.DataFrame({'a': [5, 6], 'b': [7, 8]}),
    #    ]
    #    self.test_cudf_dataframe = cudf.DataFrame({'a': [9, 10], 'b': [11, 12]})
    #    self.test_pd_dataframe = pd.DataFrame({'a': [13, 14], 'b': [15, 16]})
    #    self.test_parquet_filepath = 'test_file.parquet'
    #    self.test_csv_filepath = 'test_file.csv'

    #    self.test_cudf_dataframe.to_parquet(self.test_parquet_filepath)
    #    self.test_cudf_dataframe.to_csv(self.test_csv_filepath, index=False, header=True)

    # def tearDown(self):
    #    if os.path.exists(self.test_parquet_filepath):
    #        os.remove(self.test_parquet_filepath)


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_memory_storage(storage_type, file_format):
    dm = DataManager(storage_type='in_memory')
    assert (len(dm) == 0)

    sid = dm.store(test_cudf_dataframe)

    assert (len(dm) == 1)
    assert (sid in dm)


def test_filesystem_storage_type():
    dm = DataManager(storage_type='filesystem')
    assert (len(dm) == 0)


def test_invalid_storage_type():
    dm = DataManager(storage_type='invalid')

    assert (dm.storage_type == 'in_memory')


def test_add_remove_source():
    dm = DataManager(storage_type='in_memory')
    new_source = pd.DataFrame({'a': [9, 10], 'b': [11, 12]})

    sid = dm.store(new_source)
    assert (len(dm) == 1)
    dm.remove(sid)
    assert (len(dm) == 0)


def test_filesystem_storage_files_exist():
    dm = DataManager(storage_type='filesystem')
    sid1 = dm.store(test_cudf_dataframe)
    sid2 = dm.store(test_pd_dataframe)

    files = dm.source
    for file_path in files:
        assert (os.path.exists(file_path))

    dm.remove(sid1)
    dm.remove(sid2)

    files = dm.source
    for file_path in files:
        assert (not os.path.exists(file_path))


def test_large_fileset_filesystem_storage():
    num_dataframes = 100
    dataframes = [cudf.DataFrame({'a': [i, i + 1], 'b': [i + 2, i + 3]}) for i in range(num_dataframes)]
    dm = DataManager(storage_type='filesystem')

    source_ids = [dm.store(df) for df in dataframes]
    assert (len(dm) == num_dataframes)

    for source_id in source_ids:
        assert (source_id in dm)

    files = dm.source
    for file_path in files:
        assert (os.path.exists(file_path))

    for source_id in source_ids:
        dm.remove(source_id)

    assert (len(dm) == 0)

    files = dm.source
    for file_path in files:
        assert (not os.path.exists(file_path))


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_load_cudf_dataframe(storage_type, file_format):
    dm = DataManager(storage_type=storage_type, file_format=file_format)
    sid = dm.store(test_cudf_dataframe)
    loaded_df = dm.load(sid)

    pd.testing.assert_frame_equal(
        loaded_df.to_pandas(), test_cudf_dataframe.to_pandas())


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_load_pd_dataframe(storage_type, file_format):
    dm = DataManager(storage_type=storage_type, file_format=file_format)
    sid = dm.store(test_pd_dataframe)
    loaded_df = dm.load(sid)

    pd.testing.assert_frame_equal(loaded_df.to_pandas(), test_pd_dataframe)


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_load(storage_type, file_format):
    dm = DataManager(storage_type=storage_type, file_format=file_format)
    sid = dm.store(test_cudf_dataframe)
    loaded_df = dm.load(sid)

    pd.testing.assert_frame_equal(
        loaded_df.to_pandas(), test_cudf_dataframe.to_pandas())


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_load_non_existent_source_id(storage_type, file_format):
    dm = DataManager(storage_type=storage_type, file_format=file_format)

    try:
        dm.load(uuid.uuid4())
        pytest.fail('Expected KeyError to be raised. (Source ID does not exist.')
    except KeyError:
        pass


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_get_num_rows(storage_type, file_format):
    dm = DataManager(storage_type=storage_type, file_format=file_format)
    sid = dm.store(test_pd_dataframe)
    num_rows = dm.get_num_rows(sid)
    assert (num_rows == len(test_pd_dataframe))


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_source_property(storage_type, file_format):
    dm = DataManager(storage_type=storage_type, file_format=file_format)
    sid = dm.store(test_cudf_dataframe)
    sources = dm.source
    assert (len(sources) == 1)

    if (storage_type == 'in_memory'):
        assert (isinstance(sources[0][0], io.BytesIO))
    elif (storage_type == 'filesystem'):
        assert (isinstance(sources[0][0], str))


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_store_from_existing_file_path(storage_type, file_format):
    dm = DataManager(storage_type=storage_type, file_format=file_format)
    if (file_format == 'parquet'):
        sid = dm.store(test_parquet_filepath)
    elif (file_format == 'csv'):
        sid = dm.store(test_csv_filepath)

    loaded_df = dm.load(sid)
    assert (loaded_df.equals(test_cudf_dataframe))


if __name__ == '__main__':
    unittest.main()
