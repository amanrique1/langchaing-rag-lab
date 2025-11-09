import json
from typing import List
from pathlib import Path
import shutil
from application.ports.chunk_store import ChunkStore
from domain.models.chunk import Chunk

DEFAULT_OUTPUT_DIR = "./output_chunks"

class FileSystemChunkStore(ChunkStore):
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, chunks: list[Chunk]):
        for chunk in chunks:
            file_path = (
                self.output_dir / f"chunk_{chunk.metadata.get('chunk_index', 0)}.json"
            )
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                    },
                    f,
                    indent=4,
                )

    def get(self, chunk_id: str) -> Chunk | None:
        file_path = self.output_dir / f"chunk_{chunk_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return Chunk(content=data["content"], metadata=data["metadata"])
        return None

    def delete(self, chunk_id: str):
        file_path = self.output_dir / f"chunk_{chunk_id}.json"
        if file_path.exists():
            file_path.unlink()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        raise NotImplementedError("Semantic search is not supported by FileSystemChunkStore.")

    def clear(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

