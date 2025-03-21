"""Main Module for training and evaluation of models."""
from dotenv import load_dotenv
import copy

import src.eval.eval_saved_model as eval_saved_model
import wandb
from src.trainer import Trainer
from src.util.util import (load_config_yaml, load_config_yaml_and_wandb,
                           wandb_start)

load_dotenv()


def main():
    r"""Main."""
    config = load_config_yaml_and_wandb(None)
    jobs = config.get("jobs", [])

    trainer = Trainer(config, None)
    if "train" in jobs:
        trainer.job_train()
    if "archive" in jobs:
        trainer.job_archive()
    if "eval" in jobs:
        trainer.job_evaluate()
    print("finished.")
    wandb.finish()


# make sure that the current config, whos keys might be overwritten by multimain, has this label.
CONFIG_BASE_LBL = "BASE"


def multimain():
    r"""Main."""
    base_config = load_config_yaml(None)
    for overrideConfig in multimain_paramlist:
        try:
            #
            config = copy.deepcopy(base_config)
            assert config["run_label"] == CONFIG_BASE_LBL
            for key, value in overrideConfig.items():
                config[key] = value
            config = wandb_start(config=config)

            jobs = config.get("jobs", [])
            trainer = Trainer(config, None)
            if "train" in jobs:
                trainer.job_train()
            if "archive" in jobs:
                trainer.job_archive()
            if "eval" in jobs:
                trainer.job_evaluate()

            print("finished.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            wandb.finish()


if __name__ == "__main__":
    # check for --multimain
    import sys

    if "--multi" in sys.argv:
        multimain()
    elif "--eval" in sys.argv:
        eval_saved_model.eval_saved()
    elif "--multi_eval" in sys.argv:
        eval_saved_model.multi_eval_model()
    else:
        main()


multimain_paramlist = [
    {
        "run_label": "firstrun",
        "layer_type": "pool",
        "dataset": "vg",
        "dataset.base_embedding.using": "clip",
        "pool_type": "mean",
    },
    {
        "run_label": "secondrun",
        "layer_type": "pool",
        "dataset": "vg",
        "dataset.base_embedding.using": "clip",
        "pool_type": "max",
    },
]
