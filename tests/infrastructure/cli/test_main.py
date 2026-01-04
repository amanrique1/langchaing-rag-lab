import pytest
import sys
from unittest.mock import patch, MagicMock, call
from src.infrastructure.cli import main
from src.domain.models.enums import LengthBasedChunkingMode, SemanticChunkingThresholdType
from src.domain.models.config_classes import ChunkingConfig, QueryConfig


@patch("src.infrastructure.cli.main.run_chunking")
def test_main_save_task(mock_run_chunking):
    """Tests that the main function calls run_chunking for the save task."""
    with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
        mock_parse_args.return_value = MagicMock(
            task="save",
            source="data",
            strategy="semantic",
            config="{}",
            local_dir="output_chunks",
            db_collection="default_collection",
            clean=False,
            dual_collection=True # Added default arg
        )
        main.main()
        mock_run_chunking.assert_called_once()


@patch("src.infrastructure.cli.main.run_talk")
def test_main_talk_task(mock_run_talk):
    """Tests that the main function calls run_talk for the talk task."""
    with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
        mock_parse_args.return_value = MagicMock(
            task="talk",
            query="test query",
            top_k=5,
            local_dir="output_chunks",
            db_collection="default_collection",
            candidates=20, # Added default
            rerank=True, # Added default
            llm_reranking=False, # Added default
            dual_collection=True # Added default
        )
        main.main()
        mock_run_talk.assert_called_once()


@patch("src.infrastructure.cli.main.run_search")
def test_main_search_task(mock_run_search):
    """Tests that the main function calls run_search for the search task."""
    with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
        mock_parse_args.return_value = MagicMock(
            task="search",
            query="test query",
            top_k=5,
            local_dir="output_chunks",
            db_collection="default_collection",
            candidates=20, # Added default
            rerank=True, # Added default
            llm_reranking=False, # Added default
            dual_collection=True # Added default
        )
        main.main()
        mock_run_search.assert_called_once()


@patch("src.infrastructure.cli.main.clean_storage")
def test_main_clean_task(mock_clean_storage):
    """Tests that the main function calls clean_storage for the clean task."""
    with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
        mock_parse_args.return_value = MagicMock(
            task="clean",
            local_dir="output_chunks",
            db_collection="default_collection",
            dual_collection=True # Added default
        )
        main.main()
        mock_clean_storage.assert_called_once()


def test_run_chunking_with_length_based_mode_conversion():
    """Tests run_chunking converts mode string to LengthBasedChunkingMode enum."""
    mock_chunking_use_case = MagicMock()
    mock_storage_use_case = MagicMock()
    mock_chunking_use_case.execute.return_value = [MagicMock()]
    
    mock_container = MagicMock()
    mock_container.get_chunking_use_case.return_value = mock_chunking_use_case
    mock_container.get_storage_use_case.return_value = mock_storage_use_case
    
    chunk_config = ChunkingConfig(
        source_path="test_path",
        strategy="length_based",
        strategy_config={"mode": "character", "chunk_size": 100}
    )
    
    main.run_chunking(chunk_config, mock_container, collection_name="test")
    
    # Verify that execute was called with the enum
    call_args = mock_chunking_use_case.execute.call_args
    assert call_args[0][1] == "length_based"
    assert call_args[0][2]["mode"] == LengthBasedChunkingMode.CHARACTER


def test_run_chunking_with_invalid_length_based_mode():
    """Tests run_chunking raises ValueError for invalid length_based mode."""
    mock_chunking_use_case = MagicMock()
    mock_storage_use_case = MagicMock()
    
    mock_container = MagicMock()
    mock_container.get_chunking_use_case.return_value = mock_chunking_use_case
    mock_container.get_storage_use_case.return_value = mock_storage_use_case
    
    chunk_config = ChunkingConfig(
        source_path="test_path",
        strategy="length_based",
        strategy_config={"mode": "invalid_mode"}
    )
    
    with pytest.raises(ValueError, match="Invalid 'mode' for length_based strategy"):
        main.run_chunking(chunk_config, mock_container, collection_name="test")


def test_run_chunking_with_semantic_threshold_conversion():
    """Tests run_chunking converts threshold_mode to SemanticChunkingThresholdType enum."""
    mock_chunking_use_case = MagicMock()
    mock_storage_use_case = MagicMock()
    mock_chunking_use_case.execute.return_value = [MagicMock()]
    
    mock_container = MagicMock()
    mock_container.get_chunking_use_case.return_value = mock_chunking_use_case
    mock_container.get_storage_use_case.return_value = mock_storage_use_case
    
    chunk_config = ChunkingConfig(
        source_path="test_path",
        strategy="semantic",
        strategy_config={"threshold_mode": "percentile"}
    )
    
    main.run_chunking(chunk_config, mock_container, collection_name="test")
    
    # Verify that execute was called with the enum
    call_args = mock_chunking_use_case.execute.call_args
    assert call_args[0][1] == "semantic"
    assert call_args[0][2]["threshold_mode"] == SemanticChunkingThresholdType.PERCENTILE


def test_run_chunking_with_invalid_semantic_threshold():
    """Tests run_chunking raises ValueError for invalid semantic threshold type."""
    mock_chunking_use_case = MagicMock()
    mock_storage_use_case = MagicMock()
    
    mock_container = MagicMock()
    mock_container.get_chunking_use_case.return_value = mock_chunking_use_case
    mock_container.get_storage_use_case.return_value = mock_storage_use_case
    
    chunk_config = ChunkingConfig(
        source_path="test_path",
        strategy="semantic",
        strategy_config={"threshold_mode": "invalid_type"}
    )
    
    with pytest.raises(ValueError, match="Invalid 'threshold_mode' for semantic strategy"):
        main.run_chunking(chunk_config, mock_container, collection_name="test")


def test_run_talk():
    """Tests run_talk function execution."""
    mock_talk_use_case = MagicMock()
    mock_talk_use_case.execute.return_value = "test answer"
    
    mock_container = MagicMock()
    mock_container.get_talk_use_case.return_value = mock_talk_use_case
    
    talk_config = QueryConfig(query="test query", top_k=5)
    
    with patch("builtins.print") as mock_print:
        main.run_talk(talk_config, mock_container, collection_name="test")
        mock_talk_use_case.execute.assert_called_once_with("test query", 5, 20, use_reranking=True)
        assert any("test query" in str(call) for call in mock_print.call_args_list)
        assert any("test answer" in str(call) for call in mock_print.call_args_list)


def test_run_search_with_results():
    """Tests run_search function with results."""
    mock_search_use_case = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "test content"
    mock_chunk.metadata = {"source": "test.md"}
    mock_chunk.score = 0.95
    mock_search_use_case.execute.return_value = [mock_chunk]
    
    mock_container = MagicMock()
    mock_container.get_search_use_case.return_value = mock_search_use_case
    
    talk_config = QueryConfig(query="test query", top_k=5)
    
    with patch("builtins.print") as mock_print:
        main.run_search(talk_config, mock_container, collection_name="test")
        mock_search_use_case.execute.assert_called_once_with(query="test query", top_k=5, num_candidates=20, use_reranking=True)
        assert any("Found 1 relevant chunks" in str(call) for call in mock_print.call_args_list)


def test_run_search_without_results():
    """Tests run_search function without results."""
    mock_search_use_case = MagicMock()
    mock_search_use_case.execute.return_value = []
    
    mock_container = MagicMock()
    mock_container.get_search_use_case.return_value = mock_search_use_case
    
    talk_config = QueryConfig(query="test query", top_k=5)
    
    with patch("builtins.print") as mock_print:
        main.run_search(talk_config, mock_container, collection_name="test")
        assert any("No relevant chunks found" in str(call) for call in mock_print.call_args_list)


def test_clean_storage():
    """Tests clean_storage function."""
    mock_storage_use_case = MagicMock()
    
    mock_container = MagicMock()
    mock_container.get_storage_use_case.return_value = mock_storage_use_case
    
    with patch("builtins.print") as mock_print:
        main.clean_storage(mock_container, collection_name="test")
        mock_storage_use_case.clear.assert_called_once()
        assert any("Clearing storage" in str(call) for call in mock_print.call_args_list)
        assert any("Storage cleared successfully" in str(call) for call in mock_print.call_args_list)


@patch("src.infrastructure.cli.main.clean_storage")
@patch("src.infrastructure.cli.main.run_chunking")
def test_main_save_with_clean_flag(mock_run_chunking, mock_clean_storage):
    """Tests that clean_storage is called when --clean flag is set."""
    with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
        mock_parse_args.return_value = MagicMock(
            task="save",
            source="data",
            strategy="semantic",
            config="{}",
            local_dir="output_chunks",
            db_collection="default_collection",
            clean=True,
            dual_collection=True # Default
        )
        main.main()
        mock_clean_storage.assert_called_once()
        mock_run_chunking.assert_called_once()


def test_main_invalid_json_config():
    """Tests that invalid JSON config raises ValueError."""
    with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
        mock_parse_args.return_value = MagicMock(
            task="save",
            source="data",
            strategy="semantic",
            config="{invalid json}",
            local_dir="output_chunks",
            db_collection="default_collection",
            clean=False,
            dual_collection=True
        )
        with pytest.raises(SystemExit) as exc_info:
            main.main()
        assert exc_info.value.code == 1


def test_main_delete_task():
    """Tests that delete task prints not implemented message."""
    with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
        mock_parse_args.return_value = MagicMock(
            task="delete",
            local_dir="output_chunks",
            db_collection="default_collection",
        )
        with patch("builtins.print") as mock_print:
            main.main()
            assert any("Delete functionality is not yet implemented" in str(call) for call in mock_print.call_args_list)


