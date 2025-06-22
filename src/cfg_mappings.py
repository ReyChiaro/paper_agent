from dataclasses import dataclass


@dataclass
class ExtractorConfigs:

    model_name: str
    temperature: float
    prompt_file: str
    num_pdf_concurrent: int
    output_dir: str


@dataclass
class Configs:

    api_key: str
    base_url: str

    embed_name: str
    # extract_model_name: str
    model_name: str

    history_window: int
    rag_chunk: int
    rag_overlap: int
    rag_store: str
    rag_embed_dim: int
    rag_topk: int

    prompt_dir: str
    init_prompt_dir: str
    conversations: str
    output_dir: str
    meta_file: str
    index_file: str

    extractor: ExtractorConfigs
