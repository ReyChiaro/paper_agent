from dataclasses import dataclass


@dataclass
class ExtractorConfigs:

    temperature: float
    prompt_file: str
    num_pdf_concurrent: int
    output_dir: str


@dataclass
class RAGConfigs:

    num_chunks: int
    overlap: int
    store_dir: str
    embedding_model: str
    meta_file: str
    embed_file: str
    embed_dim: int
    topk: int


@dataclass
class Configs:

    api_key: str
    base_url: str
    model_name: str
    embed_name: str
    history_window: int
    prompt_dir: str
    init_prompt_dir: str
    conversations: str
    output_dir: str
    meta_file: str
    embed_file: str

    extractor: ExtractorConfigs
    rag: RAGConfigs
