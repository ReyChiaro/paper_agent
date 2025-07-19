import os
import logging

from pathlib import Path
from openai import OpenAI

from src.singleton import singleton
from src.debug_utils import variable_check

@singleton
class Launcher:
    """
    This project is relied on `.meta.json`, which is a metadata file that contains information about the project structure / documents information / rag information. Ensure that the project directory structure is as follows:

    ```sh
    - .
        - configs/
        - prompts/
            - .meta.json
        - rag/
            - .meta.json
        - outputs/
            - .meta.json
            - paper-1/
                - image-1.png
                - image-2.png
                - paper-1.md
                - .meta.json
            - paper-2/
    ```
    """

    def __init__(
        self,
        meta_file: str = ".meta.json",
        prompt_dir: str = "prompts",
        rag_dir: str = "rag",
        output_dir: str = "outputs",
        api_key: str = "",
        base_url: str = "",
        chat_model: str = "",
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.root = Path(os.getcwd())
        self.meta_file = Path(meta_file)
        self.prompt_dir = Path(prompt_dir)
        self.rag_dir = Path(rag_dir)
        self.output_dir = Path(output_dir)
        self._init_env()

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.chat_model = chat_model
        self.system_prompts = self._load_system_prompts()

    def _init_env(self) -> None:
        """
        Check if the project directory structure is correct. If not, create the necessary directories and files.
        """
        for directory in [self.prompt_dir, self.rag_dir, self.output_dir]:
            if not directory.exists():
                self.logger.warning(
                    f"Directory {directory} does not exist. Creating it."
                )
                directory.mkdir(parents=True)

        for directory in [self.prompt_dir, self.rag_dir, self.output_dir]:
            meta_path = directory / self.meta_file
            if not meta_path.exists():
                self.logger.warning(
                    f"Metadata file {meta_path} does not exist. Creating it."
                )
                meta_path.touch()
        self.logger.info(f"Launcher running environment initialized.")

    def _load_system_prompts(self) -> str:
        if not self.prompt_dir.exists():
            raise RuntimeError(
                f"Prompt files {self.prompt_dir} not found, please create the prompts directory and add prompts in markdown format."
            )
        prompt_files = self.prompt_dir.glob("*.md")
        system_prompts = ""
        for p in prompt_files:
            with open(p) as f:
                system_prompts += f.read()
                system_prompts += "\n"
        if not system_prompts:
            self.logger.warning(f"system_prompts are loaded but are empty.")
        return system_prompts

    def chat_single_round(self) -> None:
        variable_check(system_prompt=self.system_prompts)
        messages = [{"role": "system", "content": self.system_prompts}]
        user_inputs = input("User: ")
        messages.append({"role": "user", "content": user_inputs})
        responses = self.client.chat.completions.create(
            messages=messages,
            model=self.chat_model,
        )
        agent_outputs = f"Agent: {responses.choices[0].message.content}"
        print(agent_outputs)
