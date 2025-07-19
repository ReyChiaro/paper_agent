import json

from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Union, Any

from src.logger import get_logger
from src.paper_rag import PaperRAG
from src.pdf_extractor import PDFExtractor
from src.types.agent_info import (
    AgentInputs,
    AgentOutputs,
    Conversation,
)
from src.types.agent_info import ExtractorOutput
from src.cfg_mappings import Configs
from src.singleton import singleton


@singleton
class Controller:
    """
    Controller class is singleton class used to check data types and manage the logic flow between different components of the agent.
    """

    def __init__(
        self,
        cfgs: Configs,
        extractor: PDFExtractor,
        rag: PaperRAG,
        chat_id: Optional[str] = None,
    ) -> None:

        self.cfgs = cfgs
        self.logger = get_logger(__name__)

        # Conversation manage
        self.win_size = self.cfgs.history_window
        if chat_id is None:
            chat_id = self._init_chat()
        self.chat_id = chat_id
        self.chat_file = f"chat-{self.chat_id}.json"
        self.separator = "\n"
        self.logger.info(f"Chat initialized with ID: {self.chat_id}")

        # File manage
        self.output_dir = Path(self.cfgs.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conversation_dir = Path(self.cfgs.conversations)
        self.conversation_dir.mkdir(parents=True, exist_ok=True)

        self.meta_file = self.cfgs.meta_file
        self._pdf2meta = self._load_pdf2meta()

        self.logger.info(f"Document indices loaded")

        self.rag = rag
        self.extractor = extractor
        self.logger.info(f"Models initialized")

    def _load_pdf2meta(self) -> None:
        if (self.output_dir / self.cfgs.meta_file).exists():
            with open(self.output_dir / self.cfgs.meta_file, "r") as f:
                self._pdf2meta = json.load(f)
        else:
            self._pdf2meta = {}

    def _store_pdf2meta(self) -> None:
        with open(self.output_dir / self.meta_file, "w") as f:
            json.dump(self._pdf2meta, f, indent=4)

    def _load_document_meta(self) -> dict[str, str]:
        pass

    def _store_document_meta(self, extractor_output: ExtractorOutput) -> None:
        meta_file = extractor_output.save_dir / self.meta_file
        with open(meta_file, "w") as f:
            json.dump(asdict(extractor_output), f, indent=4)

    def _init_chat(self) -> str:
        chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
        return chat_id

    def _save_pdf_in_markdown(self, pdf_path: Path) -> ExtractorOutput:
        """
        Convert PDF to markdown and store the metadata.
        """
        output = self.extractor.convert_pdf_to_markdown(pdf_path)
        self._pdf2meta[output.pdf_path] = output.markdown_name
        self._store_document_meta(output)
        return output

    def _load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt from the init prompt directory.
        """
        prompt_path = Path(self.cfgs.init_prompt_dir) / f"{prompt_name}.md"
        return prompt_path.read_text()

    def _force_refresh_local_data(self, files: list[Path] = None) -> None:
        files = self._pdf2meta.keys() if files is None else files
        for file in files:
            output = self._save_pdf_in_markdown(file)
            markdown_path = output.save_dir / output.markdown_name
            self.rag.vectorization_persistent(markdown_path, output.pdf_name)
        self._store_pdf2meta()
        self._load_pdf2meta()

    def _rag_search(self, query_texts: str) -> list[tuple[float, dict[str, str]]]:
        return self.rag.search(query_texts)

    def _search_results_to_query(
        self,
        search_results: list[tuple[float, dict[str, str]]],
    ) -> str:
        """
        `search_results`
            list(tuple(match_score: float, chunk_info: {"filename": str, "chunk": str}))
        """
        contents = "## Reference Documents\n\n"
        for i, (score, chunk_info) in enumerate(search_results):
            contents += f"### Document {i + 1}\n\n"
            contents += (
                f"**From**: {chunk_info['filename']}\n"
                + f"**Match score**: {score}\n\n"
                + f"**Content**: {chunk_info['chunk']}\n\n"
            )
        return contents

    def preprocess_agent_inputs(
        self,
        agent_inputs: AgentInputs,
        multiround: bool = False,
        enable_rag: bool = False,
        force_refresh: bool = False,
    ) -> AgentInputs:
        texts = agent_inputs.texts
        files = agent_inputs.files

        # Force refresh the markdowns and vector stores
        if force_refresh:
            # Re-store the converted markdowns
            # Override the existing (or not) markdowns based on provided files
            # this will ignore all existing markdowns and vector stores
            self._force_refresh_local_data(files)

        sys_prompts = self._load_prompt(f"_sys_prompts{'_rag' if enable_rag else ''}")
        query = [{"role": "system", "content": sys_prompts}]

        if multiround:
            history = ""
            pass

        if enable_rag:
            # TODO search based on provided files only or all files stored
            search_results = self.rag_search(texts)
            rag_contents = self._search_results_to_query(search_results)
            query.append({"role": "user", "content": rag_contents})

        query.append({"role": "user", "content": texts})
        agent_inputs.query = query

    def _load_history_conversations(self) -> list[Conversation]:
        history_file = self.conversation_dir / self.chat_file
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
        history_file = self.conversation_dir / self.chat_file
        with open(history_file, "a") as f:
            f.write(json_str + self.separator)
        return history_file

    def _convert_conversations_to_message(
        self,
        conversations: list[Conversation],
    ) -> tuple[list[dict[str, str]], list[Path]]:
        contents = []
        files = set()
        for conv in conversations:
            contents.append({"role": conv.role, "content": conv.content})
            files.update(conv.file_refs)
        return contents, [Path(file) for file in files]

    def _save_extractor_output(
        self,
        extractor_output: ExtractorOutput,
    ) -> None:
        meta_file = extractor_output.save_dir / self.cfgs.meta_file
        meta_data = {
            "filename": extractor_output.pdf_name,
            "paper_title": extractor_output.paper_title,
            "normalized_title": extractor_output.normalized_title,
            "save_dir": str(extractor_output.save_dir),
            "num_images": extractor_output.num_images,
            "images": list(extractor_output.images),
            "pdf": str(extractor_output.save_dir / extractor_output.pdf_name),
        }
        with open(meta_file, "w") as f:
            json.dump(meta_data, f, indent=4)

    def _update_file_meta_map(
        self,
        pdf_name: Path,
        meta_file: Path,
    ) -> None:
        self._pdf2meta[pdf_name] = str(meta_file)

    def _write_file_meta_map(self) -> None:
        with open(Path(self.cfgs.output_dir) / self.cfgs.index_file, "w") as f:
            json.dump(self._pdf2meta, f, indent=4)

    def _store_file_in_markdown(
        self,
        file_paths: Union[list[Path], Path],
    ) -> ExtractorOutput:
        if isinstance(file_paths, Path):
            file_paths = [file_paths]
        # Update the stored markdowns in the disk
        results: list[ExtractorOutput] = [
            self.extractor.convert_pdf_to_markdown(f) for f in file_paths
        ]
        for res in results:
            # Update meta file in the disk
            self._save_extractor_output(res)
            # Update the file indices in the memory not the disk
            self._update_file_meta_map(
                pdf_name=res.pdf_name,
                meta_file=res.save_dir / self.cfgs.meta_file,
            )
        # Write the file indices to the disk
        self._write_file_meta_map()
        return results

    def _store_markdown_in_rag(
        self,
        pdf_markdown_maps: dict[str, Path],
    ) -> tuple[Path]:
        """
        Store markdown into RAG vector store, no matter if the markdowns are already stored in the vector store, this method will always refresh the vector store with the new markdowns. So make sure the markdowns are filtered before calling.
        """
        vector_store_path = self.rag.get_vector_store_meta_path()
        vector_store_meta_path = self.rag.get_vector_store_meta_path()
        if vector_store_path.exists() and vector_store_meta_path.exists():
            self.rag.load_index()
            self.rag.load_meta()
        self.rag.vectorize_markdowns(pdf_markdown_maps)
        return vector_store_path, vector_store_meta_path

    def _load_document_chunks(
        self,
        query_texts: str,
    ) -> list[dict[str, Any]]:
        try:
            self.rag.load_index()
            self.rag.load_meta()
            id2cid = self.rag.id2chunk
            query_embeds = self.rag.embed(query_texts)
            _, ids = self.rag.search(query_embeds)
            print(ids)
            chunks = []
            for cid in ids[0]:
                if cid in id2cid:
                    uuid_key = id2cid[cid]
                    chunks.append(self.rag._meta[uuid_key])
            return chunks
        except Exception as e:
            self.logger.warning(f"Failed to load document chunks: {e}")
            return []

    def _convert_rag_chunks_to_message(
        self,
        rag_chunks: list[dict[str, Any]],
    ) -> dict[str, str]:
        doc = "## Reference Documents\n\n"
        for i, chunk in enumerate(rag_chunks):
            doc += f"### Document {i + 1}: {chunk['paper']}\n\n"
            doc += f"{chunk['chunk']}\n\n"
        contents = {"role": "system", "content": doc}
        return contents

    def _preprocess(
        self,
        agent_inputs: AgentInputs,
        force_refresh: bool = False,
        multiround: bool = False,
        enable_rag: bool = False,
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
        files: list[Path] = agent_inputs.files
        texts: str = agent_inputs.texts
        enable_rag: bool = enable_rag

        if force_refresh:
            self.logger.info(
                f"`force_refresh` is enabled, all files "
                f"will be re-extractied into markdowns and "
                f"re-stored into vector stores."
            )
            self.logger.info(f"Following files will be refreshed: {files}")
            extractor_out = self._store_file_in_markdown(files)
            self._store_markdown_in_rag(
                {out.paper_title: out.save_dir / out.save_name for out in extractor_out}
            )

        # Detect the repeated files
        candidate_files = [f for f in files if f.name not in self._pdf2meta.keys()]
        if candidate_files:
            self.logger.info(f"Following files will be newly stored: {files}")
            extractor_out = self._store_file_in_markdown(candidate_files)
            self._store_markdown_in_rag(
                {out.paper_title: out.save_dir / out.save_name for out in extractor_out}
            )

        self.logger.info(
            f"Files are all extracted/stored:"
            f"\nMarkdown: {self.cfgs.output_dir}"
            f"\nRAG: {self.cfgs.rag.store_dir}"
        )

        if not texts:
            enable_rag = False
            if not files:
                texts = open(Path(self.cfgs.init_prompt_dir) / "_greetings.md").read()
            else:
                texts = open(Path(self.cfgs.init_prompt_dir) / "_summary.md").read()

        sys_prompts = open(
            Path(self.cfgs.init_prompt_dir)
            / f"_sys_prompts{'_rag' if enable_rag else ''}.md"
        ).read()
        agent_inputs.query = [{"role": "system", "content": sys_prompts}]

        if enable_rag:
            self.logger.info("Using RAG to retrieve relevant documents")
            import traceback

            try:
                rag_chunks = self._load_document_chunks(texts)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(e)
            self.logger.info(f"{len(rag_chunks)} found in vector store")
            if rag_chunks:
                rag_contents = self._convert_rag_chunks_to_message(rag_chunks)
                agent_inputs.query.extend(rag_contents)
            else:
                agent_inputs.query.append(
                    {"role": "system", "content": "No relevant documents found."}
                )

        if multiround:
            conversations = self._load_history_conversations()
            contents, refs = self._convert_conversations_to_message(
                conversations[-self.win_size * 2 :]
                if len(conversations) > self.win_size * 2
                else conversations
            )
            agent_inputs.query.extend(contents)
            agent_inputs.files.extend(refs)

        agent_inputs.query.append({"role": "user", "content": texts})
        agent_inputs.files.extend(files)
        agent_inputs.files = list[set(agent_inputs.files)]

        return agent_inputs
