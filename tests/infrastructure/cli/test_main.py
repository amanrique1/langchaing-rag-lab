import pytest
from unittest.mock import patch, MagicMock
from src.infrastructure.cli import main
# Import the actual classes for isinstance checks
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore
from src.infrastructure.adapters.chunk_stores.file_system_chunk_store import FileSystemChunkStore


@pytest.fixture(autouse=True)
def cleanup_mocks():
    # Restore the original classes after each test
    yield
    main.ChromaChunkStore = ChromaChunkStore
    main.FileSystemChunkStore = FileSystemChunkStore

@patch('argparse.ArgumentParser.parse_args')
@patch('src.infrastructure.cli.main.run_chunking')
def test_main_save_task(mock_run_chunking, mock_parse_args):
    # Test saving to ChromaDB
    mock_args = MagicMock(
        task='save', source='some/path', strategy='semantic', config='{}',
        local_dir='output_chunks', chroma_collection='test_collection', clean=False
    )
    mock_parse_args.return_value = mock_args
    main.main()
    mock_run_chunking.assert_called()

    # Test saving to FileSystem
    mock_args = MagicMock(
        task='save', source='some/path', strategy='semantic', config='{}',
        local_dir='output', chroma_collection='default_collection', clean=False
    )
    mock_parse_args.return_value = mock_args
    main.main()
    mock_run_chunking.assert_called()

@patch('argparse.ArgumentParser.parse_args')
@patch('src.infrastructure.cli.main.StorageUseCase')
def test_main_clean_task_chroma(mock_storage_use_case, mock_parse_args):
    mock_parse_args.return_value = MagicMock(
        task='clean', local_dir='output_chunks', chroma_collection='test_collection'
    )
    main.main()
    mock_storage_use_case.return_value.clear.assert_called_once()

@patch('argparse.ArgumentParser.parse_args')
@patch('src.infrastructure.cli.main.StorageUseCase')
def test_main_clean_task_filesystem(mock_storage_use_case, mock_parse_args):
    mock_parse_args.return_value = MagicMock(
        task='clean', local_dir='output', chroma_collection='default_collection'
    )
    main.main()
    mock_storage_use_case.return_value.clear.assert_called_once()

@patch('src.infrastructure.cli.main.ChunkingUseCase')
@patch('src.infrastructure.cli.main.StorageUseCase')
def test_run_chunking(mock_storage_use_case, mock_chunking_use_case):
    mock_chunk_config = MagicMock()
    main.run_chunking(mock_chunk_config, mock_chunking_use_case, mock_storage_use_case)
    mock_chunking_use_case.execute.assert_called_once()
    mock_storage_use_case.save.assert_called_once()

# You can add more tests for other tasks (search, delete) and error conditions
@patch('argparse.ArgumentParser.parse_args')
@patch('src.infrastructure.cli.main.run_search')
def test_main_search_task(mock_run_search, mock_parse_args):
    mock_parse_args.return_value = MagicMock(task='search', query='test query', top_k=5)
    main.main()
    mock_run_search.assert_called_once()

@patch('argparse.ArgumentParser.parse_args')
def test_main_delete_task(mock_parse_args, capsys):
    mock_parse_args.return_value = MagicMock(task='delete')
    main.main()
    captured = capsys.readouterr()
    assert "Delete functionality is not yet implemented." in captured.out

@patch('argparse.ArgumentParser.parse_args')
def test_main_save_no_source(mock_parse_args):
    mock_parse_args.return_value = MagicMock(task='save', source=None, config='{}')
    with pytest.raises(TypeError):
        main.main()