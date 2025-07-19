from typing import Optional, Union, List
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


@dataclass
class CLISchema:

    paper: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Path to the paper or a list of paper paths."},
    )


@dataclass
class LauncherConfig:

    meta_file: str = field(
        default=".meta.json",
        metadata={"help": "Metadata file name."},
    )
    prompt_dir: str = field(
        default="prompts",
        metadata={"help": "Directory for prompts."},
    )
    rag_dir: str = field(
        default="rag",
        metadata={"help": "Directory for RAG (Retrieval-Augmented Generation) data."},
    )
    output_dir: str = field(
        default="outputs",
        metadata={"help": "Directory for outputs."},
    )
    api_key: str = field(
        default="",
        metadata={"help": "API key for the OpenAI client."},
    )
    base_url: str = field(
        default="",
        metadata={"help": "Base URL for the OpenAI client."},
    )
    chat_model: str = field(
        default="",
        metadata={"help": "Chat model to be used by the OpenAI client."},
    )


@dataclass
class AgentConfigs:

    cli: CLISchema = field(default_factory=CLISchema)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)


cs = ConfigStore.instance()
cs.store(name="configs/config_schema_test", node=AgentConfigs)
