import src._force_single_process

import os
import hydra
import torch
import traceback

from pathlib import Path
from hydra.utils import instantiate

from src.pdf_extractor import PDFExtractor
from src.paper_rag import PaperRAG
from src.cfg_mappings import Configs
from src.controller import Controller
from src.types.agent_info import AgentInputs


@torch.inference_mode()
@hydra.main(config_path="configs", config_name="configs", version_base="v1.2")
def main(configs: Configs):

    pdf_extractor = PDFExtractor(instantiate(configs.extractor))
    rag = PaperRAG(instantiate(configs.rag))

    controller = Controller(
        configs,
        extractor=pdf_extractor,
        rag=rag,
        chat_id=None,
    )

    inputs = AgentInputs(
        files=[Path("documents") / p for p in os.listdir(f"documents")],
        query=[],
        texts="What is the titles of these papers?",
    )

    print(inputs)
    inputs = controller.preprocess(
        inputs,
        force_refresh=True,
        multiround=False,
        enable_rag=True,
    )
    print(inputs)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        exit(1)
