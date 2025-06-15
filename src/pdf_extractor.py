import os
import openai
import asyncio
import aiofiles
import json

from openai import AsyncOpenAI
from cfg_mappings import ExtractorConfigs
from pathlib import Path


class PDFExtractor:

    def __init__(
        self,
        extractor_cfg: ExtractorConfigs,
        api_key: str,
        base_url: str,
        pdf_paths: list[Path],
    ) -> None:

        self.cfg: ExtractorConfigs = extractor_cfg
        self.model_name: str = self.cfg.model_name
        self.temperature: float = self.cfg.temperature
        self.prompt: str = open(self.cfg.prompt_file).read()
        
        self.pdf_paths: list[Path] = pdf_paths
        self._pdf_path_exists()
        
        self._aclient = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.semaphore = asyncio.Semaphore(self.cfg.num_pdf_concurrent)

    def _pdf_path_exists(self) -> bool:
        for pdf_path in self.pdf_paths:
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        return True
    
    async def _upload_pdf(
        self,
        pdf_path: Path,
    ) -> str:
        """
        Ref: https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2712576.html&renderType=iframe

        NOTE: For current version, only qwen-long is supported for pdf-extraction.
        """
        async with self.semaphore:
            file = await self._aclient.files.create(
                file=pdf_path,
                purpose="file-extract",
            )
        return file.id
    
    async def _extract_title(
        self,
        pdf_id: str,
    ) -> str:
        repsonse = await self._aclient.chat.completions.create(
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


import hydra
from hydra.utils import instantiate
from cfg_mappings import Configs
@hydra.main(config_path="../configs", config_name="configs", version_base="v1.2")
def main(configs: Configs):
    async def demo(configs: Configs):
        api_key = configs.api_key
        base_url = configs.base_url
        pdf_paths = [
            Path(f"{os.getcwd()}/../documents/{f}") for f in os.listdir("../documents") if f.endswith(".pdf")
        ]
        print(pdf_paths)
        extractor_cfg = instantiate(configs.extractor)
        extractor = PDFExtractor(
            extractor_cfg=extractor_cfg,
            api_key=api_key,
            base_url=base_url,
            pdf_paths=pdf_paths,
        )
        pdf_ids = [await extractor._upload_pdf(pdf_path) for pdf_path in extractor.pdf_paths]
        titles = [await extractor._extract_title(pdf_id) for pdf_id in pdf_ids]
        print(titles)
    asyncio.run(demo(configs))

if __name__ =="__main__":
    main()

        

    