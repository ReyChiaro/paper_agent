import hydra

from src import AgentConfigs
from hydra.utils import instantiate


@hydra.main(version_base="v1.2", config_path="configs", config_name="config_test")
def main(cfgs: AgentConfigs):
    print(cfgs)
    launcher = instantiate(cfgs.launcher)


if __name__ == "__main__":
    main()
