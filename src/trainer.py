import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LinearLR, SequentialLR, StepLR
from torch_geometric.loader import DataLoader
from tqdm import trange
from wandb.sdk.wandb_config import Config

import wandb
from src.data_handling.datasethandler import rie_ds
from src.data_handling.sg_parser import load_raw_train_val_test
from src.eval.model_evaluation import eval
from src.eval.statistics import RetrievalMetric, SimilarityMetric
from src.models.component_handler import (contrastive_loss_from_config,
                                          loss_from_config, model_from_config)
from src.util.util import get_data_dims, guarantee_dir, logme
from src.util.device import DEVICE


class Trainer:

    def __init__(self, config, load_model_name=None):
        r"""
        @param config: dict or Config object

        sets:
        - self.config
        - self.model
        - self.scheduler
        - self.optimizer
        - self.loss_fn: fn
        - self.contrastive_loss_fn: fn|None
        """
        print(f"Data Directory: {os.getenv("DATA_DIR")}")

        self.config = config

        # initialize seed to make run reproducible.
        seed = self.config["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        self._init_data()
        self._init_model(load_model_name)
        self._init_stats()

        # init loss
        self.loss_fn = loss_from_config(self.config)
        self.contrastive_loss_fn = contrastive_loss_from_config(
            self.config
        )  # might be None.

    def _init_model(self, load_model_name):
        # model
        # data has to be initialized to init the model!
        input_dim, output_dim, edge_attr_dim = get_data_dims(self.train_ds[0])
        self.model = model_from_config(
            self.config, input_dim, output_dim, edge_attr_dim
        )

        if load_model_name is not None:
            print(f"info: loading model from: {load_model_name}")
            self.model.load_state_dict(
                torch.load(f"out/models/{load_model_name}"))
            self.model.eval()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.epochs = self.config["epochs"]
        debug_mode = self.config.get("run_label") == "DEBUG"
        if debug_mode:
            self.epochs = 10
            print("DEBUG MODE: epochs set to 10.")
            # print("DEBUG_MODE: SET no warmup")
            # self.config["lr_warmup"] = False
        warmup_epochs = int(self.epochs * 0.05)
        main_step_size = int(
            self.epochs / self.config.get("lr_step_count", 10))

        if not self.config.get("lr_warmup", False):
            self.scheduler = StepLR(
                optimizer=self.optimizer,
                step_size=main_step_size,
                gamma=self.config.get("lr_decay", 0.5),
            )
        else:
            warmup_scheduler = LinearLR(
                optimizer=self.optimizer, start_factor=0.1, total_iters=warmup_epochs
            )
            main_scheduler = StepLR(
                optimizer=self.optimizer,
                step_size=main_step_size,
                gamma=self.config.get("lr_decay", 0.5),
            )
            self.scheduler = SequentialLR(
                optimizer=self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )

        print(self.model)

    def _init_data(self):
        # 1. Initialize
        # initialize dataset
        (self.raw_train_data, self.raw_val_data, self.raw_test_data) = (
            load_raw_train_val_test(
                self.config,
                [],
            )
        )

        self.train_ds = rie_ds(self.raw_train_data, self.config, True)
        self.val_ds = rie_ds(self.raw_val_data, self.config, True)
        self.test_ds = rie_ds(self.raw_test_data, self.config)

        def data_loader(ds):
            batch_size = self.config["batch_size"]
            num_workers = self.config["xspeed.num_workers"]
            prefetch_factor = self.config["xspeed.prefetch_factor"]

            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
            )

        self.train_loader = data_loader(self.train_ds)
        self.val_loader = data_loader(self.val_ds)
        self.test_loader = data_loader(self.test_ds)

    def _init_stats(self):
        r"""
        After initializing model and data, initialize statistics.
        """
        self.stats = [
            SimilarityMetric(
                self.config, self.model, self.val_ds[:100], "estimate_val_"
            ),
            SimilarityMetric(
                self.config, self.model, self.train_ds[:100], "estimate_train_"
            ),
            RetrievalMetric(self.config, self.model, self.val_ds, "val"),
            RetrievalMetric(self.config, self.model,
                            self.train_ds[:1000], "est_train"),
        ]

    def job_train(self):
        r"""
        trains model. returns newly trained model.
        """
        # run training!
        epoch_range = trange(self.epochs, unit="epoch")
        for epoch in epoch_range:
            # train
            self.train(self.train_loader)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            # stats
            (cmd_desc, stats) = self.testing_output(
                epoch,
                self.train_loader,
                self.val_loader,
            )
            epoch_range.set_description(cmd_desc)
            if self.config.get("livestats", True) and (
                epoch % 10 == 0 or epoch in [1, 2, 3, 5]
            ):
                self.model.eval()
                with torch.no_grad():
                    for s in self.stats:
                        stats.update(s.run())
            wandb.log(stats)

        return self.model

    def job_archive(self):
        guarantee_dir("out/models")
        guarantee_dir("out/configs")
        guarantee_dir("out/splits")
        # save model and config. Saved config needed for evaluation scripts.
        if not self.config.get("OFFLINE", False):
            stamp = f"{datetime.now().strftime("%Y%m%d_%H%M%S")}_{self.config["dataset.base_embedding.using"]}_{self.config["dataset"]}_{self.config.get("run_label")}"

            torch.save(self.model.state_dict(), f"out/models/{stamp}")
            with open(f"out/configs/{stamp}", "w") as f:
                # fix for case of wandb config.
                if isinstance(self.config, Config):
                    dict_conf = self.config.as_dict()
                else:
                    dict_conf = self.config
                yaml.dump(dict_conf, f)
            print(f"Model saved as {stamp}")
            with open(f"out/splits/{stamp}_train", "w") as f:
                for g in self.train_ds:
                    f.write(f"{g["file_name"]}\n")
            with open(f"out/splits/{stamp}_val", "w") as f:
                for g in self.val_ds:
                    f.write(f"{g["file_name"]}\n")
            with open(f"out/splits/{stamp}_test", "w") as f:
                for g in self.test_ds:
                    f.write(f"{g["file_name"]}\n")
        else:
            print("WARNING: Offline, model not saved.")

    def job_evaluate(self):
        seed = self.config["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        logme(eval(self))
        if self.config.get("eval.also_on_train_data", False):
            logme(eval(self, True))
        # if not self.config.get("OFFLINE", False):
        #     if self.config["dataset"] == "psg":
        #         # print clip trio pca to console.
        #         print(f"---------------- Results for trio_pca ({self.config.get("run_label", "")}) ----------------")
        #         clip_trio_pca(self, max_data = 10)
        #         clip_trio_pca(self)
        #         print("--------------------------(end of Results for trio_pca)-------------------------")
        # else:
        #     print("WARNING: Offline, PCA evaluation not done.")

    def testing_output(self, epoch, train_loader, val_loader):
        loss = self.config.get("loss", "<LOSS>")

        train_score, contrastive_loss_train = self.evaluate(
            train_loader
        )
        test_score, contrastive_loss_test = self.evaluate(
            val_loader)

        wandb_stats = {
            f"loss/{loss}/train": train_score,
            f"loss/{loss}/val": test_score,
            f"lr": self.scheduler.get_last_lr()[0],
        }
        if self.contrastive_loss_fn is not None:
            wandb_stats[f"loss/{loss}/contrastive-train"] = contrastive_loss_train
            wandb_stats[f"loss/{loss}/contrastive-val"] = contrastive_loss_test

        cmd_description = f"Epoch: {epoch:03d}, Train {loss}: {train_score:.2f} +c {(contrastive_loss_train):.2f}, Val {loss}: {test_score:.2f} +a+c {(contrastive_loss_test):.2f}, lr:{self.scheduler.get_last_lr()[0]:.5f}"

        return (cmd_description, wandb_stats)

    def batch_loss(self, batch):
        r"""
        @returns (loss, contrastive_loss)
        """
        batch = batch.to(DEVICE)

        out = self.model(batch)
        # loss rule: left: image, right: graph
        loss = self.loss_fn(batch.y.view_as(out), out).cpu()
        c_loss = 0
        if self.contrastive_loss_fn is not None:
            c_loss = self.contrastive_loss_fn(batch.y.view_as(out), out).cpu()
        return (loss, c_loss)

    def evaluate(self, loader):
        self.model.eval()
        with torch.no_grad():
            loss_tuples = [self.batch_loss(b) for b in loader]
            losses, contrastive_losses = map(
                list, zip(*loss_tuples))
            return (np.mean(losses), np.mean(contrastive_losses))

    def train(self, train_loader):
        self.model.train()
        lambda_normal = self.config.get("loss_lambda", 1)
        lambda_contrastive = self.config.get("contrastive_loss_lambda", 1)

        for b in train_loader:
            self.optimizer.zero_grad()
            (loss, contrastive_loss) = self.batch_loss(b)
            (
                lambda_normal * loss
                + lambda_contrastive * contrastive_loss
            ).backward()
