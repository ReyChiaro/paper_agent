import re
import threading

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from PIL import Image
from pathlib import Path
from rich.live import Live
from src.logger import get_logger, beautified_tqdm
from src.cfg_mappings import ExtractorConfigs
from src.client import get_client


class PDFExtractor:

    def __init__(
        self,
        pdf_paths: list[Path],
        extractor_cfgs: ExtractorConfigs,
    ):
        self.logger = get_logger(__name__)
        self.client = get_client()

        self.cfg: ExtractorConfigs = extractor_cfgs

        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pdf_paths: list[Path] = pdf_paths

        configs = {
            "output_format": "markdown",
            "output_dir": self.output_dir,
            "use_llm": False,
            "workers": 0,
        }
        config_parser = ConfigParser(configs)
        self.pdf_converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

    def check_pdf_paths(
        self,
    ) -> None:
        for p in self.pdf_paths:
            if not p.exists():
                raise FileNotFoundError(f"PDF file not found: {p}")

    def extract_pdf_title(
        self,
        markdown_text: str,
    ) -> list[str]:
        return re.findall(r"^# (.+)$", markdown_text, re.MULTILINE)[0]

    def normalize_title(
        self,
        title: str,
    ) -> str:
        title = re.sub(r"[^A-Za-z0-9 ]+", "", title)
        title = title.lower()
        title = re.sub(r"\s+", "-", title)
        return title

    def save_images(
        self,
        image: Image.Image,
        path_to_save: Path,
    ) -> None:
        try:
            image.save(path_to_save)
        except Exception as e:
            self.logger.warning(f"Failed to save image {path_to_save}: {e}")

    def convert_pdf_to_markdown(
        self,
        pdf_path: Path,
    ) -> tuple[Path, str, str]:

        self.logger.info(f"Start `marker` to convert PDF: {pdf_path}")
        with beautified_tqdm():
            rendered = self.pdf_converter(str(pdf_path))
            markdown_text, _, images = text_from_rendered(rendered)
                
        self.logger.info(f"Converting finished")

        title = self.extract_pdf_title(markdown_text)
        normalized_title = self.normalize_title(title)
        normalized_title = normalized_title[:50]
        self.logger.info(f"Paper title: {title}, normalized title: {normalized_title}")

        save_dir = self.output_dir / normalized_title
        # conflict_count = 1
        # while save_dir.exists():
        #     save_dir = self.output_dir / f"{normalized_title}-{conflict_count}"
        #     conflict_count += 1
        save_dir.mkdir(parents=True, exist_ok=True)

        [
            self.save_images(image, save_dir / path_to_save)
            for path_to_save, image in images.items()
        ]

        with open(save_dir / f"{normalized_title}.md", "w") as md:
            md.write(markdown_text)
        self.logger.info(f"Markdown files and images saved to {save_dir}")

        return save_dir, title, normalized_title

    def files_repeat_check(
        self,
        files: list[Path],
    ) -> list[Path]:
        server_files = self.client.files.list().model_dump()["data"]
        existed_files = {info["filename"]: info["id"] for info in server_files}
        candidate_files = set(f.name for f in files)
        repeat_files = candidate_files.intersection(existed_files.keys())
        if repeat_files:
            self.logger.info(
                f"Following files already exist on the server, they will not be uploaded:"
            )
            for f in repeat_files:
                self.logger.info(f"- {f} (id: {existed_files[f]})")
        return [f for f in files if f.name not in repeat_files]

    def upload_file(
        self,
        file_path: str,
    ) -> str:
        file_info = self.client.files.create(file=file_path, purpose="file-extract")
        return file_info.id
