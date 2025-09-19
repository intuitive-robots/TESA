import argparse
import copy
import os

import torch

import wandb
from src.eval.model_evaluation import clip_trio_eval, clip_trio_pca
from src.eval.statistics import SimilarityMetric
from src.trainer import Trainer
from src.util.util import (load_config_yaml, load_config_yaml_and_wandb,
                           overwrite_config, wandb_start)


DATA_BASE = os.getenv("DATA_DIR")


def eval_saved():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="bool for specifiying if we want to evaluate")
    parser.add_argument("--model", type=str, help="specify model to use")
    parser.add_argument("--similarity", action="store_true",
                        help="metric evaluation")
    parser.add_argument("--trio_eval", action="store_true",
                        help="eval on (I,T,G)")
    parser.add_argument(
        "--trio_pca",
        type=str,
        default=None,
        help="graph|image|text|combined",
    )
    parser.add_argument(
        "--text_prefix",
        type=str,
        default="",
        help="prefix for embedding texts for trio_eval / trio_pca",
    )
    parser.add_argument(
        "--max_data", type=int, default=1000, help="max data for trio-pca"
    )
    parser.add_argument(
        "--on_coco_captions", default=False, help="use coco captions for trio eval"
    )
    parser.add_argument(
        "--export_plot", default=False, help="export plot for trio pca to /out/figs"
    )
    args = parser.parse_args()
    overconfig = load_config_yaml(None, "config_eval.yaml")
    config = overwrite_config(
        load_config_yaml_and_wandb(args.model), overconfig)
    print(f"eval-only using eval_saved_model.py, on {args.model}")
    trainer = Trainer(config, args.model)

    any = args.eval or args.similarity or args.trio_eval or args.trio_pca
    if args.eval or not any:
        trainer.job_evaluate()  # sgg on test (I,G)
    if args.similarity:
        sim = SimilarityMetric(config, trainer.model, trainer.test_ds[:200])
        print(sim.run())
    if args.trio_eval:
        print(
            clip_trio_eval(
                trainer,
                args.text_prefix,
                args.model,
                on_coco_captions=args.on_coco_captions,
            )
        )  # sgg on eval. (I,T,G)
    if args.trio_pca:
        clip_trio_pca(
            args.model,
            trainer,
            args.trio_pca,
            args.text_prefix,
            args.max_data,
            on_coco_captions=args.on_coco_captions,
            export_plot=args.export_plot,
        )  # pca on eval. (I,T,G)
    print(
        f"done ({config["dataset"]}, {config["dataset.base_embedding.using"]})")


def multi_eval_model():
    r"""Main."""
    preloaded_configs = []
    for mpl in multimain_paramlist:
        preloaded_configs.append(
            load_config_yaml(mpl["m"], "config_eval.yaml"))

    for base_config, overrideConfig in zip(preloaded_configs, multimain_paramlist):
        try:
            model = overrideConfig["m"]
            del overrideConfig["m"]
            config = copy.deepcopy(base_config)
            for key, value in overrideConfig.items():
                config[key] = value
            config = wandb_start(config=config)
            trainer = Trainer(config, model)
            trainer.job_evaluate()

            print("finished.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            wandb.finish()


multimain_paramlist = [
    {
        "m": "20250223_200700_SigLIP_psg_-1",
        "eval.qa.statements": "naive",
        "run_label": "qa-PSgn",
        "eval.base_embeddings": "graph",
    },
]
