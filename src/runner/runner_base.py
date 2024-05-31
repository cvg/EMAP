import torch
import torch.nn.functional as F
import os
import numpy as np
from shutil import copyfile
from icecream import ic
from pyhocon import ConfigFactory, HOCONConverter
from src.dataset.dataset import Dataset
from src.models.udf_model import (
    UDFNetwork,
    BetaNetwork,
    SingleVarianceNetwork,
)
from src.models.udf_renderer_blending import UDFRendererBlending
from src.models.loss import EdgeLoss


class Runner:
    def __init__(
        self,
        conf,
        mode="train",
        is_continue=False,
        args=None,
    ):
        # Initial setting
        self.device = torch.device("cuda")
        self.conf = conf

        self.base_exp_dir = os.path.join(
            self.conf["general.base_exp_dir"],
            str(self.conf["dataset"]["scan"]),
            self.conf["general.expname"],
        )
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset = Dataset(self.conf["dataset"])
        self.near, self.far = self.dataset.near, self.dataset.far

        self.iter_step = 0

        # trainning parameters
        self.end_iter = self.conf.get_int("train.end_iter")
        self.save_freq = self.conf.get_int("train.save_freq")
        self.report_freq = self.conf.get_int("train.report_freq")
        self.val_freq = self.conf.get_int("train.val_freq")
        self.batch_size = self.conf.get_int("train.batch_size")
        self.validate_resolution_level = self.conf.get_int(
            "train.validate_resolution_level"
        )
        self.use_white_bkgd = self.conf.get_bool("train.use_white_bkgd")
        self.importance_sample = self.conf.get_bool("train.importance_sample")

        # setting about learning rate schedule
        self.learning_rate = self.conf.get_float("train.learning_rate")
        self.learning_rate_geo = self.conf.get_float("train.learning_rate_geo")
        self.learning_rate_alpha = self.conf.get_float("train.learning_rate_alpha")
        self.warm_up_end = self.conf.get_float("train.warm_up_end", default=0.0)
        self.anneal_end = self.conf.get_float("train.anneal_end", default=0.0)
        # don't train the udf network in the early steps
        self.fix_geo_end = self.conf.get_float("train.fix_geo_end", default=200)
        self.warmup_sample = self.conf.get_bool(
            "train.warmup_sample", default=False
        )  # * training schedule
        # whether the udf network and appearance network share the same learning rate
        self.same_lr = self.conf.get_bool("train.same_lr", default=False)

        # weights
        self.igr_weight = self.conf.get_float("train.igr_weight")
        self.igr_ns_weight = self.conf.get_float("train.igr_ns_weight", default=0.0)

        # loss functions
        self.edge_loss_func = EdgeLoss(self.conf["edge_loss"]["loss_type"])
        self.edge_weight = self.conf.get_float("edge_loss.edge_weight", 0.0)
        self.is_continue = is_continue
        # self.is_finetune = args.is_finetune

        self.mode = mode
        self.model_type = self.conf["general.model_type"]
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        params_to_train_nerf = []
        params_to_train_geo = []

        self.nerf_outside = None
        self.nerf_coarse = None
        self.nerf_fine = None
        self.sdf_network_fine = None
        self.udf_network_fine = None
        self.variance_network_fine = None

        # self.nerf_outside = NeRF(**self.conf["model.nerf"]).to(self.device)
        self.udf_network_fine = UDFNetwork(**self.conf["model.udf_network"]).to(
            self.device
        )
        self.variance_network_fine = SingleVarianceNetwork(
            **self.conf["model.variance_network"]
        ).to(self.device)
        self.beta_network = BetaNetwork(**self.conf["model.beta_network"]).to(
            self.device
        )
        # params_to_train_nerf += list(self.nerf_outside.parameters())
        params_to_train_geo += list(self.udf_network_fine.parameters())
        params_to_train += list(self.variance_network_fine.parameters())
        params_to_train += list(self.beta_network.parameters())

        self.optimizer = torch.optim.Adam(
            [
                {"params": params_to_train_geo, "lr": self.learning_rate_geo},
                {"params": params_to_train},
                {"params": params_to_train_nerf},
            ],
            lr=self.learning_rate,
        )

        self.renderer = UDFRendererBlending(
            self.nerf_outside,
            self.udf_network_fine,
            self.variance_network_fine,
            self.beta_network,
            device=self.device,
            **self.conf["model.udf_renderer"],
        )

    def update_learning_rate(self, start_g_id=0):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (
                self.end_iter - self.warm_up_end
            )
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                1 - alpha
            ) + alpha

        for g in self.optimizer.param_groups[start_g_id:]:
            g["lr"] = self.learning_rate * learning_factor

    def update_learning_rate_geo(self):
        if self.iter_step < self.fix_geo_end:  # * make bg nerf learn first
            learning_factor = 0.0
        elif self.iter_step < self.warm_up_end * 2:
            learning_factor = self.iter_step / (self.warm_up_end * 2)
        elif self.iter_step < self.end_iter * 0.5:
            learning_factor = 1.0
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.end_iter * 0.5) / (
                self.end_iter - self.end_iter * 0.5
            )
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                1 - alpha
            ) + alpha

        for g in self.optimizer.param_groups[:1]:
            g["lr"] = self.learning_rate_geo * learning_factor

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def train(self):
        self.train_udf()

    def get_flip_saturation(self, flip_saturation_max=0.9):
        start = 10000
        if self.iter_step < start:
            flip_saturation = 0.0
        elif self.iter_step < self.end_iter * 0.5:
            flip_saturation = flip_saturation_max
        else:
            flip_saturation = 1.0

        return flip_saturation

    def file_backup(self):
        # copy python file
        dir_lis = self.conf["general.recording"]
        cur_dir = os.path.join(self.base_exp_dir, "recording")
        os.makedirs(cur_dir, exist_ok=True)
        files = os.listdir("./")
        for f_name in files:
            if f_name[-3:] == ".py":
                copyfile(os.path.join("./", f_name), os.path.join(cur_dir, f_name))

        for dir_name in dir_lis:
            os.system(f"cp -r {dir_name} {cur_dir}")

        # copy configs
        # copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))
        with open(
            os.path.join(self.base_exp_dir, "recording", "config.conf"), "w"
        ) as fd:
            res = HOCONConverter.to_hocon(self.conf)
            fd.write(res)

    def train_udf(self):
        return NotImplementedError

    def load_checkpoint(self, checkpoint_name):
        return NotImplementedError

    def save_checkpoint(self):
        return NotImplementedError

    def validate(self, idx=-1, resolution_level=-1):
        return NotImplementedError
