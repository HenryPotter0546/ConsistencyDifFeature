import argparse
from omegaconf import OmegaConf

def get_wandb_run_name(config):
    timestep_list = "_".join(str(x) for x in config["save_timestep"])
    model_name = config["lcm_model_name"]
    sample_method = config["extract_mode"]
    wandb_run = f"timesteps_{timestep_list}_{model_name}_{sample_method}"
    return wandb_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/train.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    config = OmegaConf.to_container(config, resolve=True)

    print(get_wandb_run_name(config))