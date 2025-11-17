import pytest
from unittest.mock import patch, MagicMock
from src.infrastructure.cli import main


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
            chroma_collection="default_collection",
            clean=False,
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
            chroma_collection="default_collection",
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
            chroma_collection="default_collection",
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
            chroma_collection="default_collection",
        )
        main.main()
        mock_clean_storage.assert_called_once()
