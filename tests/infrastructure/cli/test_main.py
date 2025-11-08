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
@patch('src.infrastructure.cli.main.ChromaChunkStore')
@patch('src.infrastructure.cli.main.FileSystemChunkStore')
def test_main_save_task(mock_file_store, mock_chroma_store, mock_run_chunking, mock_parse_args):
    # Test saving to ChromaDB
    mock_parse_args.return_value = MagicMock(
        task='save', source='some/path', local=False, collection_name='test_collection',
        strategy='semantic', config='{}', loader_mode='single'
    )
    main.main()
    mock_chroma_store.assert_called_with('test_collection')
    mock_run_chunking.assert_called()

    # Test saving to FileSystem
    mock_parse_args.return_value = MagicMock(
        task='save', source='some/path', local=True, output_dir='output',
        strategy='semantic', config='{}', loader_mode='single'
    )
    main.main()
    mock_file_store.assert_called_with('output')
    mock_run_chunking.assert_called()

@patch('argparse.ArgumentParser.parse_args')
@patch('src.infrastructure.cli.main.ChromaChunkStore', autospec=True)
def test_main_clean_task_chroma(mock_chroma_store_class, mock_parse_args):
    main.ChromaChunkStore = mock_chroma_store_class
    mock_store_instance = MagicMock(spec=ChromaChunkStore)
    mock_chroma_store_class.return_value = mock_store_instance
    
    mock_parse_args.return_value = MagicMock(
        task='clean', local=False, collection_name='test_collection', output_dir='output_chunks'
    )
    
    main.main()
    
    mock_store_instance.clear.assert_called_once()

@patch('argparse.ArgumentParser.parse_args')
@patch('src.infrastructure.cli.main.FileSystemChunkStore', autospec=True)
def test_main_clean_task_filesystem(mock_file_store_class, mock_parse_args):
    main.FileSystemChunkStore = mock_file_store_class
    mock_store_instance = MagicMock(spec=FileSystemChunkStore)
    mock_file_store_class.return_value = mock_store_instance

    mock_parse_args.return_value = MagicMock(
        task='clean', local=True, output_dir='output', collection_name='default_collection'
    )

    main.main()
    
    mock_store_instance.clear.assert_called_once()


@patch('src.infrastructure.cli.main.MarkdownDocumentLoader')
@patch('src.infrastructure.cli.main.ChunkingUseCase')
def test_run_chunking(mock_use_case, mock_loader):
    mock_args = MagicMock(
        source='some/path', strategy='semantic', config='{}', loader_mode='single'
    )
    mock_chunk_store = MagicMock()
    
    main.run_chunking(mock_args, mock_chunk_store)

    mock_loader.assert_called_once()
    mock_use_case.assert_called_once()
    mock_chunk_store.save.assert_called_once()

def test_setup_common_arguments():
    parser = main.argparse.ArgumentParser()
    main.setup_common_arguments(parser)
    # Very basic check to see if arguments are added
    actions = [action.dest for action in parser._actions]
    assert 'source' in actions
    assert 'strategy' in actions
    assert 'config' in actions
    assert 'loader_mode' in actions

# You can add more tests for other tasks (search, delete) and error conditions
@patch('argparse.ArgumentParser.parse_args')
def test_main_search_task(mock_parse_args, capsys):
    mock_parse_args.return_value = MagicMock(task='search', local=False, collection_name='test_collection', output_dir='output_chunks')
    main.main()
    captured = capsys.readouterr()
    assert "Search functionality not yet implemented." in captured.out

@patch('argparse.ArgumentParser.parse_args')
def test_main_delete_task(mock_parse_args, capsys):
    mock_parse_args.return_value = MagicMock(task='delete', local=False, collection_name='test_collection', output_dir='output_chunks')
    main.main()
    captured = capsys.readouterr()
    assert "Delete functionality not yet implemented." in captured.out

@patch('argparse.ArgumentParser.parse_args')
def test_main_save_no_source(mock_parse_args, capsys):
    mock_parse_args.return_value = MagicMock(task='save', source=None, local=False, collection_name='test_collection', output_dir='output_chunks')
    main.main()
    captured = capsys.readouterr()
    assert "Error: 'source' argument is required" in captured.out