import re

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.output import text_from_rendered
from marker.models import create_model_dict
from PIL import Image
from pathlib import Path

from src.singleton import singleton
from src.logger import get_logger, beautified_tqdm
from src.cfg_mappings import ExtractorConfigs
from src.types.agent_info import ExtractorOutput


@singleton
class PDFExtractor:

    def __init__(self, extractor_cfgs: ExtractorConfigs):
        self.logger = get_logger(__name__)

        self.cfg: ExtractorConfigs = extractor_cfgs

        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
    ) -> ExtractorOutput:

        self.logger.info(f"Using `marker` to convert PDF: {pdf_path}")
        with beautified_tqdm():
            rendered = self.pdf_converter(str(pdf_path))
            markdown_text, _, images = text_from_rendered(rendered)

        self.logger.info(f"Converting finished")

        title = self.extract_pdf_title(markdown_text)
        normalized_title = self.normalize_title(title)
        normalized_title = normalized_title[:50]
        self.logger.info(f"Paper title: {title}, normalized title: {normalized_title}")

        save_dir = self.output_dir / normalized_title
        save_dir.mkdir(parents=True, exist_ok=True)

        for path_to_save, image in images.items():
            self.save_images(image, save_dir / path_to_save)

        with open(save_dir / f"{normalized_title}.md", "w") as md:
            md.write(markdown_text)

        self.logger.info(f"Markdown files and images saved to {save_dir}")

        outputs = ExtractorOutput(
            pdf_path=pdf_path,
            pdf_name=pdf_path.name,
            paper_title=title,
            normalized_title=normalized_title,
            save_dir=save_dir,
            markdown_name=f"{normalized_title}.md",
            num_images=len(images),
            images=images.keys(),
        )

        return outputs
