import uuid
import json
import faiss
import numpy as np

from typing import Union
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.singleton import singleton
from src.cfg_mappings import RAGConfigs


@singleton
class PaperRAG:

    def __init__(self, rag_configs: RAGConfigs) -> None:

        self.rag_chunk = rag_configs.rag_chunk
        self.rag_overlap = rag_configs.rag_overlap
        self.vector_store = Path(rag_configs.rag_store)
        self.vector_store.mkdir(parents=True, exist_ok=True)
        self.rag_topk = rag_configs.rag_topk

        self.meta = {}

        self.embedding_model = SentenceTransformer(rag_configs.embedding_model, device="cpu")
        self.index_model = faiss.IndexFlatL2(rag_configs.rag_embed_dim)
        self.index_id_map = faiss.IndexIDMap(self.index_model)

    @property
    def id2chunk(self):
        return {int(uuid.UUID(k)) >> 64: k for k in self.meta.keys()}

    def get_vector_store_path(self) -> Path:
        return self.vector_store / "vectors.index"

    def get_vector_store_meta_path(self) -> Path:
        return self.vector_store / "meta.json"

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

    def encode(self, texts: Union[list[str], str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.embedding_model.encode(texts).astype(np.float32)

    def search(self, query_embeds: np.ndarray):
        return self.index_id_map.search(query_embeds, self.rag_topk)

    def store_index(self) -> None:
        faiss.write_index(self.index_id_map, str(self.vector_store / "vectors.index"))

    def store_meta(self) -> None:
        meta_file = self.vector_store / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(self.meta, f, indent=4)

    def load_index(self) -> bool:
        index_file = self.vector_store / "vectors.index"
        if index_file.exists():
            self.index_id_map = faiss.read_index(str(index_file))
            return True
        return False

    def load_meta(self) -> bool:
        meta_file = self.vector_store / "meta.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                self.meta = json.load(f)
            return True
        return False

    def vectorize_markdowns(
        self,
        pdf_markdown_maps: dict[str, Path],
    ) -> None:
        for name, markdown_path in pdf_markdown_maps.items():
            chunks = self._split_paper_markdown(markdown_path)
            self._markdown_to_index(name, chunks)
        self.store_index()
        self.store_meta()
