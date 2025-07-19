import os
import logging

from pathlib import Path
from openai import OpenAI


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
    ):
        self.logger = logging.getLogger(__name__)

        self.root = Path(os.getcwd())
        self.meta_file = Path(meta_file)
        self.prompt_dir = Path(prompt_dir)
        self.rag_dir = Path(rag_dir)
        self.output_dir = Path(output_dir)
        self._init_env()
        self._load_system_prompts()

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _init_env(self):
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
    
    def _load_system_prompts(self):
        pass

    def chat(self):
        user_inputs = input("User: ")
        self.client.chat.completions.create(
            messages=
        )