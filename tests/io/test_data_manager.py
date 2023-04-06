import cudf
import io
import os
import pandas as pd
import pytest
import unittest
import uuid

from morpheus.io import data_manager

DataManager = data_manager.DataManager

sources = [
    cudf.DataFrame({'a': [1, 2], 'b': [3, 4]}),
    'buffer2.parquet',  # Local or remote file path
    pd.DataFrame({'a': [5, 6], 'b': [7, 8]}),
]
test_cudf_dataframe = cudf.DataFrame({'a': [9, 10], 'b': [11, 12]})
test_pd_dataframe = pd.DataFrame({'a': [13, 14], 'b': [15, 16]})
test_parquet_filepath = 'test_file.parquet'
test_csv_filepath = 'test_file.csv'

test_cudf_dataframe.to_parquet(test_parquet_filepath)
test_cudf_dataframe.to_csv(test_csv_filepath, index=False, header=True)


def tearDownModule():
    if os.path.exists(test_parquet_filepath):
        os.remove(test_parquet_filepath)
    if os.path.exists(test_csv_filepath):
        os.remove(test_csv_filepath)


@pytest.mark.parametrize("storage_type", ['in_memory'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_memory_storage(storage_type, file_format):
    dm = DataManager(storage_type=storage_type, file_format=file_format)
    assert (len(dm) == 0)

    sid = dm.store(test_cudf_dataframe)
    assert (len(dm) == 1)
    assert (sid in dm)


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
def test_filesystem_storage_type(storage_type):
    dm = DataManager(storage_type=storage_type)
    assert (len(dm) == 0)
    assert (dm.storage_type == storage_type)


@pytest.mark.parametrize("storage_type", ['invalid', "something else invalid"])
def test_invalid_storage_type(storage_type):
    dm = DataManager(storage_type=storage_type)

    assert (dm.storage_type == 'in_memory')


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_add_remove_source(storage_type, file_format):
    dm = DataManager(storage_type=storage_type, file_format=file_format)
    new_source = pd.DataFrame({'a': [9, 10], 'b': [11, 12]})

    sid = dm.store(new_source)
    assert (len(dm) == 1)
    dm.remove(sid)
    assert (len(dm) == 0)


@pytest.mark.parametrize("storage_type", ['filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_filesystem_storage_files_exist(storage_type, file_format):
    dm = DataManager(storage_type=storage_type, file_format=file_format)
    sid1 = dm.store(test_cudf_dataframe)
    sid2 = dm.store(test_pd_dataframe)

    files = dm.manifest
    for file_path in files.values():
        assert (os.path.exists(file_path))

    dm.remove(sid1)
    dm.remove(sid2)

    files = dm.manifest
    for file_path in files:
        assert (not os.path.exists(file_path))


@pytest.mark.parametrize("storage_type", ['filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_large_fileset_filesystem_storage(storage_type, file_format):
    num_dataframes = 100
    dataframes = [cudf.DataFrame({'a': [i, i + 1], 'b': [i + 2, i + 3]}) for i in range(num_dataframes)]
    dm = DataManager(storage_type=storage_type, file_format=file_format)

    source_ids = [dm.store(df) for df in dataframes]
    assert (len(dm) == num_dataframes)

    for source_id in source_ids:
        assert (source_id in dm)

    files = dm.manifest.values()
    for file_path in files:
        assert (os.path.exists(file_path))

    for source_id in source_ids:
        dm.remove(source_id)

    assert (len(dm) == 0)

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
    num_rows = dm.get_record(sid).num_rows
    assert (num_rows == len(test_pd_dataframe))


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_source_property(storage_type, file_format):
    dm = DataManager(storage_type=storage_type, file_format=file_format)
    sid = dm.store(test_cudf_dataframe)
    data_records = dm.records

    assert (len(data_records) == 1)

    for k, v in data_records.items():
        assert (v._storage_type == storage_type)
        if (storage_type == 'in_memory'):
            assert (isinstance(v.data, io.BytesIO))
        elif (storage_type == 'filesystem'):
            assert (isinstance(v.data, str))


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
