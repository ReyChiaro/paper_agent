import os
import hydra
import asyncio
import logging

from pathlib import Path
from hydra.utils import instantiate

from src.cfg_mappings import Configs
from src.pdf_extractor import PDFExtractor


@hydra.main(config_path="configs", config_name="configs", version_base="v1.2")
def main(configs: Configs):

    os.environ["API_KEY"] = configs.api_key
    os.environ["BASE_URL"] = configs.base_url

    async def demo(configs: Configs):
        pdf_paths = [
            Path(f"{os.getcwd()}/documents/{f}")
            for f in os.listdir("documents")
            if f.endswith(".pdf")
        ]

        extractor_cfgs = instantiate(configs.extractor)
        extractor = PDFExtractor(
            extractor_cfgs=extractor_cfgs,
            pdf_paths=pdf_paths,
        )
        results = [
            extractor.convert_pdf_to_markdown(pdf_path) for pdf_path in pdf_paths
        ]
        save_dirs, filenames, titles = zip(*results)
        print(save_dirs, filenames, titles)

        file_paths = extractor.files_repeat_check(
            [save_dir / filenames for save_dir, filenames in zip(save_dirs, filenames)]
        )
        results = [extractor.upload_file(file_path) for file_path in file_paths]
        print(results)

    asyncio.run(demo(configs))


if __name__ == "__main__":
    main()
