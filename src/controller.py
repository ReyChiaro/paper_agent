import os
import json

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.pdf_extractor import PDFExtractor
from src.types.agent_info import AgentInputs
from src.types.agent_info import Conversation
from src.types.agent_info import ExtractorOutputs
from src.cfg_mappings import Configs


class Controller:
    """
    Controller class is singleton class used to check data types and manage the logic flow between different components of the agent.
    """

    def __init__(
        self,
        cfgs: Configs,
        chat_id: Optional[str] = None,
    ) -> None:

        self.cfgs = cfgs

        # Conversation manage
        self.win_size = self.cfgs.history_window
        if chat_id is None:
            chat_id = self._init_chat()
        self.chat_id = chat_id
        self.chat_file = f"chat-{self.chat_id}.json"
        self.separator = "\n"

        # File manage
        file_indices_str = (
            open(Path(self.cfgs.output_dir) / self.cfgs.index_file)
            .read()
            .replace("\n", ",")
        )
        self.file_indices = json.loads(f"[{file_indices_str}]")

    def _init_chat(self) -> str:
        chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
        return chat_id

    def _load_history_conversations(self) -> list[Conversation]:
        history_file = Path(self.cfgs.conversations) / self.chat_file
        convs = list[Conversation] = []
        contents = open(history_file, "r").read().split(self.separator)
        for content in contents:
            convs.append(Conversation(**json.loads(content)))
        return convs

    def _store_one_conversation(
        self,
        round_id: int,
        role: str,
        content: str,
        file_refs: list[Path],
    ) -> Path:
        conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        timstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_refs = [str(file_ref) for file_ref in file_refs]
        conversation = Conversation(
            conversation_id=conversation_id,
            round_id=round_id,
            timestamp=timstamp,
            role=role,
            content=content,
            file_refs=file_refs,
        )
        json_str = json.dumps(asdict(conversation))
        history_file = Path(self.cfgs.conversations) / self.chat_file
        with open(history_file, "a") as f:
            f.write(json_str + self.separator)
        return history_file

    def _convert_conversations_to_message(
        self,
        conversations: list[Conversation],
    ) -> tuple[list[dict[str, str]], set[str]]:
        contents = []
        files = set()
        for conv in conversations:
            contents.append({"role": conv.role, "content": conv.content})
            files.update(conv.file_refs)
        return contents, files

    def _write_extractor_outputs_to_meta(
        self,
        extractor_outputs: ExtractorOutputs,
    ) -> None:
        meta_file = Path(self.cfgs.output_dir) / self.cfgs.meta_file
        meta_data = {
            "filename": extractor_outputs.filename,
            "paper_title": extractor_outputs.paper_title,
            "normalized_title": extractor_outputs.normalized_title,
            "save_dir": str(extractor_outputs.save_dir),
            "num_images": extractor_outputs.num_images,
            "images": extractor_outputs.images,
            "pdf": str(extractor_outputs.save_dir / extractor_outputs.filename),
        }
        with open(meta_file, "w") as f:
            json.dump(meta_data, f, indent=4)

    def _update_file_indices(
        self,
        pdf_file: Path,
        meta_file: Path,
    ) -> None:
        self.file_indices[pdf_file.name] = meta_file

    def _write_file_indices(self) -> None:
        with open(self.cfgs.output_dir / self.cfgs.index_file, "w") as f:
            for p, m in self.file_indices.items():
                f.write('\{"{}": "{}"\}\n'.format(p.name, str(m)))

    def preprocess(
        self,
        agent_inputs: AgentInputs,
        force_refresh: bool = False,
        multiround: bool = False,
    ) -> AgentInputs:
        """
        Preprocess the inputs of agent. This method will load prompts and concatenate the prompts with the user query, the markdowns if exist, and the history messages if `multiround=True`, to construct a complete query for the LLM.

        The inputs may contain PDF files or text queries.

        - If the PDF files are provided and they are not stored locally, this method will convert PDFs to markdown and save them. If there are stored markdowns, this methods will load them unless user ask for `force_refresh=True`.
        - If the text queries are provided, this method will invoke the LLM API to answer based on the provided markdowns (if exist), otherwise it will answer based only on the text.

        NOTE: To check if the provided PDF files are converted, check the value of the key `pdf` in `.meta` file in the output directory, this value points to the absolute path of the original PDF file.

        Args:
            agent_inputs (AgentInputs): The inputs of the agent, which may contain PDF files or text queries.
            force_refresh (bool, optional): If True, the method will always refresh the markdown, even if it is already stored locally. Defaults to False.
            multiround (bool, optional): If True, the method will load the history messages and concatenate them with the user query. Defaults to False.
        """
        files: set[str] = agent_inputs.files
        query: str = agent_inputs.query

        if force_refresh:
            extractor = PDFExtractor(extractor_cfgs=self.cfgs.extractor)
            results: list[ExtractorOutputs] = [
                extractor.convert_pdf_to_markdown(f) for f in files
            ]
            for res in results:
                self._write_extractor_outputs_to_meta(res)
                self._update_file_indices(
                    res.save_dir / res.filename,
                    res.save_dir / self.cfgs.meta_file,
                )
            self._write_file_indices()
        elif files.intersection(set(self.file_indices.keys())):
            candidate_files = [
                Path(f).name for f in files if f not in self.file_indices.keys()
            ]
            if candidate_files:
                extractor = PDFExtractor(extractor_cfgs=self.cfgs.extractor)
                results: list[ExtractorOutputs] = [
                    extractor.convert_pdf_to_markdown(Path(f)) for f in candidate_files
                ]
                for res in results:
                    self._write_extractor_outputs_to_meta(res)
                    self._update_file_indices(
                        res.save_dir / res.filename,
                        res.save_dir / self.cfgs.meta_file,
                    )
            self._write_file_indices()

        # After extracting the files,
        # all selected files map to a meta file,
        # where the converted markdown information is stored.

        if not query:
            if not files:
                query = open(Path(self.cfgs.init_prompt_dir) / "_greetings.md").read()
            else:
                query = open(Path(self.cfgs.init_prompt_dir) / "_summary.md").read()

        agent_inputs.query = [{"role": "system", "content": self.sys_prompts}]
        if multiround:
            conversations = self._load_history_conversations()
            contents, refs = self._convert_conversations_to_message(
                conversations[-self.win_size * 2 :]
                if len(conversations) > self.win_size * 2
                else conversations
            )
            agent_inputs.query.append({"role": "user", "content": contents})
            agent_inputs.files.update(refs)
        agent_inputs.query.append({"role": "user", "content": query})
        agent_inputs.files.update(files)
        return agent_inputs
