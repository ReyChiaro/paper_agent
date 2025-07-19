from dataclasses import dataclass


@dataclass
class MainConfigs:

    PROMPT_DIR: str = "prompts"
    RAG_DIR: str = "rag"
    META_FILE: str = ".meta.json"
    API_KEY: str = ""
    BASE_URL: str = ""
    CHAT_MODEL: str = ""
