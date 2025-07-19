import hydra

from src import AgentConfigs
from hydra.utils import instantiate

from src.launcher import Launcher


@hydra.main(version_base="v1.2", config_path="configs", config_name="config_test")
def main(cfgs: AgentConfigs):
    print(cfgs)
    launcher = Launcher(**cfgs.launcher)
    launcher.chat_single_round()


if __name__ == "__main__":
    main()
