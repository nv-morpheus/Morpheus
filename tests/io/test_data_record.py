import cudf
import io
import os
import pandas as pd
import pytest
import shutil
import tempfile

from morpheus.io.data_record import DataRecord

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


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_data_record_load(storage_type, file_format):
    data_record = DataRecord(data_source=test_cudf_dataframe, data_label='test_data',
                             storage_type=storage_type, file_format=file_format)
    loaded_df = data_record.load()
    pd.testing.assert_frame_equal(loaded_df.to_pandas(), test_cudf_dataframe.to_pandas())


@pytest.mark.parametrize("storage_type", ['in_memory', 'filesystem'])
@pytest.mark.parametrize("file_format", ['parquet', 'csv'])
def test_data_record_num_rows(storage_type, file_format):
    data_record = DataRecord(data_source=test_cudf_dataframe, data_label='test_data',
                             storage_type=storage_type, file_format=file_format)
    num_rows = data_record.num_rows
    assert num_rows == len(test_cudf_dataframe)


@pytest.mark.parametrize("storage_type", ['invalid', "something else invalid"])
def test_invalid_storage_type(storage_type):
    with pytest.raises(ValueError):
        DataRecord(data_source=test_cudf_dataframe, data_label='test_data',
                   storage_type=storage_type, file_format='parquet')


@pytest.mark.parametrize("file_format", ['invalid', "something else invalid"])
def test_invalid_data_format(file_format):
    with pytest.raises(ValueError):
        DataRecord(data_source=test_cudf_dataframe, data_label='test_data',
                   storage_type='in_memory', file_format=file_format)


def test_data_record_deletion_filesystem_csv():
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp_data.csv')
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_csv(temp_file)

    data_record = DataRecord(temp_file, 'test_label', 'filesystem', 'csv', copy_from_source=True)
    path_on_disk = data_record.backing_file

    del data_record
    assert not os.path.exists(path_on_disk)

    shutil.rmtree(temp_dir)


def test_data_record_deletion_filesystem_parquet():
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp_data.parquet')
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_parquet(temp_file)

    data_record = DataRecord(temp_file, 'test_label', 'filesystem', 'parquet', copy_from_source=True)
    path_on_disk = data_record.backing_file

    del data_record
    assert not os.path.exists(path_on_disk)

    shutil.rmtree(temp_dir)


def test_data_record_deletion_no_owner():
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp_data.csv')
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_csv(temp_file)

    data_record = DataRecord(temp_file, 'test_label', 'filesystem', 'csv', copy_from_source=False)
    path_on_disk = data_record.backing_file

    del data_record
    assert os.path.exists(path_on_disk)

    shutil.rmtree(temp_dir)


def test_data_record_properties_csv():
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp_data.csv')
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_csv(temp_file)

    data_record = DataRecord(temp_file, 'test_label', 'filesystem', 'csv', copy_from_source=True)

    assert data_record.backing_file == 'test_label'
    assert data_record.data == 'test_label'
    assert data_record.format == 'csv'
    assert data_record.num_rows == 3

    del data_record
    shutil.rmtree(temp_dir)


def test_data_record_properties_parquet():
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp_data.parquet')
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_parquet(temp_file)

    data_record = DataRecord(temp_file, 'test_label', 'filesystem', 'parquet', copy_from_source=True)

    assert data_record.backing_file == 'test_label'
    assert data_record.data == 'test_label'
    assert data_record.format == 'parquet'
    assert data_record.num_rows == 3

    del data_record
    shutil.rmtree(temp_dir)


def test_data_record_properties_in_memory():
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    data_record = DataRecord(df, 'test_label', 'in_memory', 'csv')

    print(data_record)
    print(repr(data_record))
    assert data_record.backing_file == 'IO Buffer'
    assert isinstance(data_record.data, io.BytesIO)
    assert data_record.format == 'csv'
    assert data_record.num_rows == 3

    del data_record
