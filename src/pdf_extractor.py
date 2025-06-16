import asyncio
import logging

from pathlib import Path

from src.cfg_mappings import ExtractorConfigs
from src.client import get_client, get_aclient


class PDFExtractor:

    def __init__(
        self,
        extractor_cfg: ExtractorConfigs,
        pdf_paths: list[Path],
    ) -> None:
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.cfg: ExtractorConfigs = extractor_cfg
        self.model_name: str = self.cfg.model_name
        self.temperature: float = self.cfg.temperature
        self.prompt: str = open(self.cfg.prompt_file).read()
        
        self.pdf_paths: list[Path] = pdf_paths
        self._pdf_path_exists()

        # Record the PDF file names and corresponding ids in this batch of PDF files
        self.pdf_metas: dict[str, str] = {k.name: "" for k in self.pdf_paths}
        
        self.client = get_client()
        self.aclient = get_aclient()
        self.semaphore = asyncio.Semaphore(self.cfg.num_pdf_concurrent)

    def _pdf_path_exists(self) -> bool:
        for pdf_path in self.pdf_paths:
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        return True
    
    def _file_repeat_check(
        self,
    ) -> list[str]:
        """
        Checks if the PDF files to be uploaded already exist in the bailian platform. If they do, it returns their IDs and updates the `self.pdf_paths` list to exclude these files.
        """
        existed_ids = []
        file_infos = self.client.files.list().model_dump()["data"]
        file_names = set([file_info["filename"] for file_info in file_infos])
        candidates = set(self.pdf_metas.keys())
        existed_files = file_names.intersection(candidates)
        updated_files = candidates - existed_files
        if existed_files:
            existed_ids = [file_info["id"] for file_info in file_infos if file_info["filename"] in existed_files]
            for name, fid in zip(existed_files, existed_ids):
                self.pdf_metas[name] = fid
            print(self.pdf_metas)
        if updated_files != set(self.pdf_paths):
            self.logger.info(f"Existing files found in bailian platform, these files will not be uploaded:")
            for file_name in existed_files:
                self.logger.info(f"\n - [name] {file_name}\n - [id] {self.pdf_metas[file_name]}")
        return existed_ids

    async def _upload_pdf(
        self,
        pdf_path: Path,
    ) -> str:
        """
        Uploads one PDF file to bailian platform, and there is no file-repeat check using bailian API, so we provided a file-repeat checking method `_file_repeat_check`. If you want to upload files, please invoke the repeat-check method. Below is the reference of bailian platform.

        Ref: 
        - https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2712576.html&renderType=iframe
        - https://help.aliyun.com/zh/model-studio/openai-file-interface?spm=0.0.0.i5

        NOTE: For current version, only qwen-long is supported for pdf-extraction.
        """
        file_name = pdf_path.name
        if not self.pdf_metas[file_name]:
            async with self.semaphore:
                file = await self.aclient.files.create(
                    file=pdf_path,
                    purpose="file-extract",
                )
                self.pdf_metas[file_name] = file.id
        return self.pdf_metas[file_name]

    async def _upload_pdfs_with_repeat_check(
        self,
        pdf_path: Path,
    ) -> str:
        existed_ids: list[str] = self._file_repeat_check()

    async def _extract_title(
        self,
        pdf_id: str,
    ) -> str:
        repsonse = await self.aclient.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "You are a helpful assistent that extracts titles from PDF documents."},
                {"role": "system", "content": f"fileid://{pdf_id}"},
                {"role": "user", "content": "Identify the title of the pdf document. Please output the title of the pdf document only, without any additional information or sentences.\n\n"
                "**Examples:**\n\nAttention Is All You Need\n\nLanguage Models are Few-Shot Learners\n\nA Survey of Large Language Models"}
            ],
            stream=False,
            n=1,
        )
        content = repsonse.choices[0].message.content.strip()
        if not content:
            raise ValueError(f"Failed to extract title from PDF with ID: {pdf_id}")
        return content

