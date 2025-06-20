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
    extract_model_name: str
    model_name: str
    output_dir: str

    extractor: ExtractorConfigs
