from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ExtractorOutputs:

    filename: str
    paper_title: str
    normalized_title: str
    save_dir: Path
    save_name: str
    num_images: int
    images: list[str]


@dataclass
class AgentInputs:

    files: set[str]
    query: str


@dataclass
class AgentOutputs:

    history: list[dict]
    query: str
    answer: str


@dataclass
class Conversation:

    conversation_id: str
    round_id: int
    timestamp: str
    role: Literal["user", "assistant"]
    content: str
    file_refs: set[str]

    # TODO: summary converstions embed contents
    summary: Optional[str] = None
    embeddings: Optional[list[float]] = None
