from pathlib import Path

from src.client import get_client
from src.types.agent_info import (
    AgentInputs,
    AgentOutputs,
    Conversation,
)


class _Agent(object):

    def __init__(
        self,
        prompt_path: Path,
    ):
        self.client = get_client()

    def basic_chat(
        self,
        pdf_paths: set[str],
        query: str,
    ) -> str:
        pass
    