import uuid
import json
import faiss
import numpy as np

from typing import Union
from pathlib import Path
from sentence_transformers import SentenceTransformer


class PaperRAG(object):

    def __init__(
        self,
        rag_chunk: int,
        rag_overlap: int,
        vector_store: str,
        embedding_model: str,
        embedding_dim: int = 384,
    ) -> None:

        self.rag_chunk = rag_chunk
        self.rag_overlap = rag_overlap
        self.vector_store = Path(vector_store)
        self.vector_store.mkdir(parents=True, exist_ok=True)

        self.meta = {}

        self.embedding_model = SentenceTransformer(embedding_model)
        self.index_model = faiss.IndexFlatL2(embedding_dim)
        self.index_id_map = faiss.IndexIDMap(self.index_model)

    @property
    def id2chunk(self):
        return {int(uuid.UUID(k)) >> 64: k for k in self.meta.keys()}

    def _split_paper_markdown(
        self,
        paper_markdown_path: Path,
    ) -> list[str]:
        with open(paper_markdown_path, "r") as f:
            contents = f.read()
        words = contents.split()
        chunks = []
        start = 0
        while start < len(words):
            chunks.append(" ".join(words[start : start + self.rag_chunk]))
            start += self.rag_chunk - self.rag_overlap
        return chunks

    def _markdown_to_index(
        self,
        name: str,
        chunks: list[str],
    ) -> None:
        ids = []
        vectors = []
        embeds = self.embedding_model.encode(chunks)

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            vectors.append(embeds[i])
            ids.append(int(uuid.UUID(chunk_id)) >> 64)
            self.meta[chunk_id] = {
                "paper": name,
                "chunk": chunk,
                "chunk_id": chunk_id,
            }

        self.index_id_map.add_with_ids(
            np.array(vectors, dtype=np.float32),
            np.array(ids, dtype=np.int64),
        )

    def store_index(self) -> None:
        faiss.write_index(self.index_id_map, self.vector_store / "vectors.index")

    def store_meta(self) -> None:
        meta_file = self.vector_store / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(self.meta, f, indent=4)

    def load_index(self) -> None:
        index_file = self.vector_store / "vectors.index"
        self.index_id_map = faiss.read_index(index_file)

    def load_meta(self) -> None:
        meta_file = self.vector_store / "meta.json"
        with open(meta_file, "r") as f:
            self.meta = json.load(f)

    def vectorize_markdowns(
        self,
        pdf_markdown_maps: dict[str, Path],
    ) -> None:
        for name, markdown_path in pdf_markdown_maps:
            chunks = self._split_paper_markdown(markdown_path)
            self._markdown_to_index(name, chunks)
        self.store_index()
        self.store_meta()
