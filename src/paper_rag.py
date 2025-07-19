"""
For this module, `faiss` is not used for both embedding and searching,
this module provides an LLM based embedding and a cos-similarity based search methods.
The embeddings are stored in vector_store, configured in `configs/`.
"""

import uuid
import json
import numpy as np

from openai import OpenAI
from typing import Union
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.singleton import singleton
from src.cfg_mappings import RAGConfigs
from src.logger import get_logger


@singleton
class PaperRAG:

    def __init__(self, cfgs: RAGConfigs, client: OpenAI) -> None:

        self.logger = get_logger(__name__)

        self._embeddings = []
        self._chunks = []
        self._ids = []

        self.num_chunks = cfgs.num_chunks
        self.overlap = cfgs.overlap
        self.topk = cfgs.topk

        self.store_dir = Path(cfgs.store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.store_dir / cfgs.meta_file
        self.embed_file = self.store_dir / cfgs.embed_file

        self.client = client
        self.embedding_name = cfgs.embedding_model
        self.embedding_dim = cfgs.embed_dim

        # Load existing metadata and embeddings if available
        if self.meta_file.exists():
            self._load_meta()
        if self.embed_file.exists():
            self._load_embeddings()

    def _load_meta(self) -> None:
        with open(self.meta_file, "r") as f:
            contents = json.load(f)
        self._ids = contents.keys()
        self._chunks = [contents[k]["chunk"] for k in self._ids]

    def _save_meta(self) -> None:
        contents = {
            k: {"chunk": self._chunks[i], "chunk_id": k}
            for i, k in enumerate(self._ids)
        }
        with open(self.meta_file, "w") as f:
            json.dump(contents, f, indent=4)

    def _load_embeddings(self) -> None:
        self._embeddings = np.load(self.embed_file).tolist()

    def _save_embeddings(self) -> None:
        np.save(self.embed_file, np.array(self._embeddings, dtype=np.float32))

    def _add(
        self,
        normalized_embedding: np.ndarray,
        chunk_info: dict[str, str],
    ) -> None:
        if np.linalg.norm(normalized_embedding) != 1:
            normalized_embedding = normalized_embedding / np.linalg.norm(
                normalized_embedding, axis=1, keepdims=True
            )
        uid = str(uuid.uuid4())
        uid = int(uuid.UUID(uid)) >> 64
        self._embeddings.append(normalized_embedding)
        self._ids.append(uid)
        self._chunks.append(chunk_info)

    def split_document(self, document_contents: str) -> list[str]:
        start = 0
        chunks = []
        words = document_contents.split()
        while start < len(words):
            chunks.append(" ".join(words[start : start + self.num_chunks]))
            start += self.num_chunks - self.overlap
        return chunks

    def embed(self, chunks: Union[list[str], str]) -> np.ndarray:
        if isinstance(chunks, str):
            chunks = [chunks]
        embeddings = []
        for chunk in chunks:
            try:
                response = self.embedding_model.embeddings.create(
                    model=self.embedding_name,
                    input=chunk,
                    dimensions=self.embedding_dim,
                    encoding_format="float",
                ).model_dump()
                embeds = response["data"][0]["embedding"]
            except Exception as e:
                embeds = [0.0] * self.embedding_dim
            finally:
                embeddings.append(embeds)
        embeddings = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.array(embeddings, dtype=np.float32)

    def search(self, query: str) -> list[tuple[float, dict[str, str]]]:
        if len(self._embeddings) == 0:
            return []
        query_embed = self.embed(query)
        embeds = np.stack(self._embeddings, axis=0)
        scores = np.dot(embeds, query_embed)
        topk_indices = np.argsort(scores)[-self.topk :][::-1]
        topk_uid = self._ids[topk_indices]
        results = [(scores[i], self._chunks[topk_uid[i]]) for i in topk_indices]
        return results

    def _vectorization(
        self,
        path: Union[str, Path],
        document_name: str,
    ) -> None:
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            self.logger.warning(f"Path {path} does not exist.")
            return
        with open(path, "r") as f:
            contents = f.read()
        chunks = self.split_document(contents)
        chunk_infos = [{"filename": document_name, "chunk": chunk} for chunk in chunks]
        embeds = self.embed(chunks)
        for embed, chunk_info in zip(embeds, chunk_infos):
            self._add(embed, chunk_info)

    def vectorization_runtime(
        self,
        path: Union[str, Path],
        document_name: str,
    ) -> None:
        self._vectorization(path, document_name)

    def vectorization_persistent(
        self,
        path: Union[str, Path],
        document_name: str,
    ) -> None:
        self._vectorization(path, document_name)
        self._save_meta()
