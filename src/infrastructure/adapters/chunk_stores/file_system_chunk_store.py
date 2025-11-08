import json
from typing import List
from pathlib import Path
from application.ports.chunk_store import ChunkStore
from domain.models.chunk import Chunk


class FileSystemChunkStore(ChunkStore):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, chunk: Chunk):
        file_path = self.output_dir / f"chunk_{chunk.metadata.get('chunk_index', 0)}.json"
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
        return None

    def delete(self, chunk_id: str):
        pass

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        return []

    def save(self, chunks: List[Chunk]):
        for chunk in chunks:
            self.add(chunk)
