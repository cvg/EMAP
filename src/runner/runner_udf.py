import torch
import torch.nn.functional as F
import os
import logging
import numpy as np
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
from icecream import ic
from tqdm import tqdm
import open3d as o3d
from termcolor import colored
from PIL import Image
import json
from src.utils import visualize_depth
from src.runner.runner_base import Runner
from src.edge_extraction.extract_pointcloud import get_pointcloud_from_udf
from src.edge_extraction.extract_parametric_edge import get_parametric_edge


class Runner_UDF(Runner):
    def __init__(
        self,
        conf,
        mode="train",
        is_continue=False,
        args=None,
    ):
        super(Runner_UDF, self).__init__(
            conf,
            mode=mode,
            is_continue=is_continue,
            args=args,
        )

    def train_udf(self):
        # Load checkpoint
        if self.is_continue:
            latest_model_name = self.conf["train.latest_model_name"]

            logging.info("Find checkpoint: {}".format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        if self.mode[:5] == "train":
            self.file_backup()

        image_perm = torch.randperm(self.dataset.n_images)
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, "logs"))

        for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate

        beta_flag = True
        psnr_list = []
        self.best_psnr = 0.0

        # tqdm with discription of loss
        par = tqdm(
            range(self.iter_step, self.end_iter),
            desc="PSNR: {:.2f}".format(0),
            position=0,
            leave=True,
        )
        for iter_i in par:
            if self.same_lr:
                self.update_learning_rate(start_g_id=0)
            else:
                self.update_learning_rate(start_g_id=1)
                self.update_learning_rate_geo()

            img_idx = image_perm[self.iter_step % len(image_perm)]
            sample = self.dataset.gen_random_rays_patches_at(
                img_idx,
                self.batch_size,
                importance_sample=self.importance_sample,
            )

            data = sample["rays"]
            pose = sample["pose"]
            intrinsics = sample["intrinsics"]
            depth_scale = sample["depth_scale"]

            # todo load supporting images

            rays_o, rays_d, true_edge = (
                data["rays_o"],
                data["rays_v"],
                data["edge"],
            )

            near, far = self.near, self.far

            mask = torch.ones_like(true_edge).float()

            mask_sum = mask.sum() + 1e-5

            render_out = self.renderer.render(
                rays_o,
                rays_d,
                near,
                far,
                depth_scale=depth_scale,
                flip_saturation=self.get_flip_saturation(),
                pose=pose,
                fx=intrinsics[0, 0],
                fy=intrinsics[1, 1],
                img_index=None,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
            )

            udf = render_out["udf"]
            edge = render_out["edge"]
            depth = render_out["depth"]

            variance = render_out["variance"]
            beta = render_out["beta"]
            gamma = render_out["gamma"]

            gradient_error = render_out["gradient_error"]
            gradient_error_near_surface = render_out["gradient_error_near_surface"]

            udf = render_out["udf"]
            udf_min = udf.min(dim=1)[0][mask[:, 0] > 0.5].mean()

            edge_loss = (
                self.edge_loss_func(
                    edge,
                    true_edge,
                )
                * self.edge_weight
            )

            psnr = 20.0 * torch.log10(
                1.0 / (((edge - true_edge) ** 2 * mask).sum() / (mask_sum)).sqrt()
            )
            psnr_list.append(psnr)

            # Eikonal loss
            gradient_error_loss = gradient_error

            if (
                variance.mean() < 2 * beta.item()
                and variance.mean() < 0.01
                and beta_flag
                and self.variance_network_fine.variance.requires_grad
            ):
                print("make beta trainable")
                self.beta_network.set_beta_trainable()
                beta_flag = False

            if (
                self.variance_network_fine.variance.requires_grad is False
                and self.iter_step > 20000
            ):
                self.variance_network_fine.set_trainable()

            igr_ns_weight = self.igr_ns_weight

            loss = (
                edge_loss
                + gradient_error_near_surface * igr_ns_weight
                + gradient_error_loss * self.igr_weight
            )

            par.set_description("PSNR: {:.2f}, Loss: {:.2f}".format(psnr, loss.item()))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar("Loss/loss", loss, self.iter_step)
            self.writer.add_scalar("Loss/edge_loss", edge_loss, self.iter_step)
            self.writer.add_scalar(
                "Loss/gradient_error_loss",
                gradient_error_loss * self.igr_weight,
                self.iter_step,
            )
            self.writer.add_scalar(
                "Loss/gradient_error_near_surface",
                gradient_error_near_surface * igr_ns_weight,
                self.iter_step,
            )
            self.writer.add_scalar("Sta/variance", variance.mean(), self.iter_step)
            self.writer.add_scalar("Sta/beta", beta.item(), self.iter_step)
            self.writer.add_scalar("Sta/psnr", psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(
                    "iter:{:8>d} loss = {:.4f} "
                    "edge_loss = {:.4f} "
                    "eki_loss = {:.4f} "
                    "eki_ns_loss = {:.4f} ".format(
                        self.iter_step,
                        loss,
                        edge_loss,
                        gradient_error_loss,
                        gradient_error_near_surface,
                    )
                )
                print(
                    "iter:{:8>d} "
                    "variance = {:.6f} "
                    "beta = {:.6f} "
                    "gamma = {:.4f} "
                    "lr_geo={:.8f} lr={:.8f} ".format(
                        self.iter_step,
                        variance.mean(),
                        beta.item(),
                        gamma.item(),
                        self.optimizer.param_groups[0]["lr"],
                        self.optimizer.param_groups[1]["lr"],
                    )
                )

                print(
                    colored(
                        "psnr = {:.4f} "
                        "weight_sum = {:.4f} "
                        "weight_sum_fg_bg = {:.4f} "
                        "udf_min = {:.8f} "
                        "udf_mean = {:.4f} "
                        "igr_ns_weight = {:.4f} "
                        "igr_weight = {:.4f} ".format(
                            psnr,
                            (render_out["weight_sum"] * mask).sum() / mask_sum,
                            (render_out["weight_sum_fg_bg"] * mask).sum() / mask_sum,
                            udf_min,
                            udf.mean(),
                            igr_ns_weight,
                            self.igr_weight,
                        ),
                        "green",
                    )
                )

                ic(self.get_flip_saturation())

            if self.iter_step % 100 == 0 and self.iter_step > 0:
                psnr_avg = sum(psnr_list) / len(psnr_list)
                psnr_list = []

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint(psnr_avg)

            if self.iter_step % self.val_freq == 0:
                self.validate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = torch.randperm(self.dataset.n_images)

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(
            os.path.join(self.base_exp_dir, "checkpoints", checkpoint_name),
            map_location=self.device,
        )
        self.udf_network_fine.load_state_dict(checkpoint["udf_network_fine"])
        self.variance_network_fine.load_state_dict(checkpoint["variance_network_fine"])
        self.beta_network.load_state_dict(checkpoint["beta_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iter_step = checkpoint["iter_step"]

        logging.info("End")

    def save_checkpoint(self, psnr_val):
        checkpoint_dir = os.path.join(self.base_exp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "udf_network_fine": self.udf_network_fine.state_dict(),
            "variance_network_fine": self.variance_network_fine.state_dict(),
            "beta_network": self.beta_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_step": self.iter_step,
        }

        if psnr_val > self.best_psnr:
            self.best_psnr = psnr_val
            logging.info(
                "Save checkpoint with the best PSNR: {:.2f} in ckpt_best.pth".format(
                    self.best_psnr
                )
            )
            best_checkpoint_path = os.path.join(checkpoint_dir, "ckpt_best.pth")
            torch.save(checkpoint, best_checkpoint_path)

    def validate(self, idx=-1, resolution_level=-1):
        logging.info("Validate begin")
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, pose, intrinsics, depth_scale = self.dataset.gen_rays_at(
            idx, resolution_level=resolution_level
        )

        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        depth_scale = depth_scale.reshape(-1, 1).split(self.batch_size)

        out_edge_fine = []
        out_edge_pixel = []
        out_normal_fine = []
        out_depth = []

        for rays_o_batch, rays_d_batch, depth_scale_batch in zip(
            rays_o, rays_d, depth_scale
        ):
            # near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            near, far = self.near, self.far

            background_edge = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                depth_scale=depth_scale_batch,
                color_maps=None,
                pose=pose,
                fx=intrinsics[0, 0],
                fy=intrinsics[1, 1],
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_edge,
            )

            feasible = lambda key: (
                (key in render_out) and (render_out[key] is not None)
            )

            if feasible("edge"):
                out_edge_fine.append(render_out["edge"].detach().cpu().numpy())
            if feasible("edge_pixel"):
                out_edge_pixel.append(render_out["edge_pixel"].detach().cpu().numpy())
            if feasible("depth"):
                out_depth.append(render_out["depth"].detach().cpu().numpy())

            if not feasible("gradients_flip"):
                if feasible("gradients") and feasible("weights"):
                    if render_out["inside_sphere"] is not None:
                        out_normal_fine.append(
                            (
                                render_out["gradients"]
                                * render_out["weights"][
                                    :,
                                    : self.renderer.n_samples
                                    + self.renderer.n_importance,
                                    None,
                                ]
                            )
                            .sum(dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        out_normal_fine.append(
                            (
                                render_out["gradients"]
                                * render_out["weights"][
                                    :,
                                    : self.renderer.n_samples
                                    + self.renderer.n_importance,
                                    None,
                                ]
                            )
                            .sum(dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
            else:
                if feasible("gradients_flip") and feasible("weights"):
                    if render_out["inside_sphere"] is not None:
                        out_normal_fine.append(
                            (
                                render_out["gradients_flip"]
                                * render_out["weights"][
                                    :,
                                    : self.renderer.n_samples
                                    + self.renderer.n_importance,
                                    None,
                                ]
                            )
                            .sum(dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        out_normal_fine.append(
                            (
                                render_out["gradients_flip"]
                                * render_out["weights"][
                                    :,
                                    : self.renderer.n_samples
                                    + self.renderer.n_importance,
                                    None,
                                ]
                            )
                            .sum(dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
            del render_out

        # image
        img_edge_fine = None
        if len(out_edge_fine) > 0:
            edge_fine = (
                np.concatenate(out_edge_fine, axis=0).reshape([H, W]) * 255
            ).clip(0, 255)
            img_edge_fine = np.array(
                Image.fromarray(edge_fine.astype(np.uint8)).convert("RGB")
            )
        img_edge_pixel = None
        if len(out_edge_pixel) > 0:
            img_edge_pixel = (
                np.concatenate(out_edge_pixel, axis=0).reshape([H, W, 3]) * 255
            ).clip(0, 255)
        os.makedirs(os.path.join(self.base_exp_dir, "edge_maps"), exist_ok=True)

        if len(out_edge_fine) > 0:
            if len(out_edge_pixel) > 0:
                edges = [img_edge_fine, img_edge_pixel]
            else:
                edges = [img_edge_fine]

            gt_edge = self.dataset.edge_at(
                idx, resolution_level=resolution_level
            ).astype(np.uint8)
            cv.imwrite(
                os.path.join(
                    self.base_exp_dir,
                    "edge_maps",
                    "{:0>8d}_{}.png".format(self.iter_step, idx),
                ),
                np.concatenate(
                    edges + [np.array(Image.fromarray(gt_edge).convert("RGB"))]
                ),
            )

        # normal
        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(
                self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy()
            )
            normal_img = (
                np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3])
                * 128
                + 128
            ).clip(0, 255)
        os.makedirs(os.path.join(self.base_exp_dir, "normals"), exist_ok=True)
        if len(out_normal_fine) > 0:
            cv.imwrite(
                os.path.join(
                    self.base_exp_dir,
                    "normals",
                    "{:0>8d}_{}.png".format(self.iter_step, idx),
                ),
                normal_img[:, :, ::-1],
            )

        # depth
        depth_vis = None
        if len(out_depth) > 0:
            pred_depth = np.concatenate(out_depth, axis=0).reshape([H, W])
            depth_final_vis = visualize_depth(pred_depth)

        os.makedirs(os.path.join(self.base_exp_dir, "depths"), exist_ok=True)
        if len(out_depth) > 0:
            cv.imwrite(
                os.path.join(
                    self.base_exp_dir,
                    "depths",
                    "{:0>8d}_{}.png".format(self.iter_step, idx),
                ),
                depth_final_vis[:, :, ::-1],
            )

    def extract_edge(
        self,
        resolution=256,
        udf_threshold=1.0,
        sampling_N=50,
        sampling_delta=5e-3,
        is_pointshift=False,
        iters=1,
        is_linedirection=False,
        visible_checking=False,
    ):
        """
        Extract edges based on the UDF network model.

        Args:
            resolution (int): Resolution of the output.
            udf_threshold (float): Threshold for UDF.
            sampling_N (int): Number of sampling points.
            sampling_delta (float): Delta for sampling.
            is_pointshift (bool): If point shift is to be applied.
            iters (int): Number of iterations for point shift.
            is_linedirection (bool): If line direction is calculated.
            visible_checking (bool): If visible checking is performed.

        Raises:
            NotImplementedError: If model type is not 'udf'.
        """
        latest_model_name = self.conf["train.latest_model_name"]
        logging.info(f"Find checkpoint: {latest_model_name}")
        self.load_checkpoint(latest_model_name)

        if self.model_type != "udf":
            raise NotImplementedError("Model type other than 'udf' is not supported.")

        func = self.udf_network_fine.udf

        def func_grad(xyz):
            gradients = self.udf_network_fine.gradient(xyz)
            gradients_mag = torch.linalg.norm(gradients, ord=2, dim=-1, keepdim=True)
            gradients_norm = gradients / (gradients_mag + 1e-5)
            return gradients_norm

        points, line_directions = get_pointcloud_from_udf(
            func,
            func_grad,
            N_MC=resolution,
            udf_threshold=udf_threshold,
            sampling_N=sampling_N,
            sampling_delta=sampling_delta,
            is_pointshift=is_pointshift,
            iters=iters,
            is_linedirection=is_linedirection,
            device=self.device,
        )

        ld_colors = (line_directions + 1) / 2.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(ld_colors)

        result_dir = os.path.join(self.base_exp_dir, "results")
        os.makedirs(result_dir, exist_ok=True)

        ply_file_path = os.path.join(result_dir, "udf_pointcloud_withdirection.ply")
        try:
            o3d.io.write_point_cloud(ply_file_path, pcd, write_ascii=True)
            logging.info(f"Saved {ply_file_path} for edge direction visualization.")
        except IOError as e:
            logging.error(f"Failed to save {ply_file_path}: {e}")

        edge_dict = {
            "resolution": resolution,
            "udf_threshold": udf_threshold,
            "points": points,
            "ld_colors": ld_colors,
            "detector": self.conf["dataset"]["detector"],
            "scene_name": self.conf["dataset"]["scan"],
            "dataset_dir": self.conf["dataset"]["data_dir"],
            "result_dir": result_dir,
        }

        pred_edge_points, return_edge_dict = get_parametric_edge(
            edge_dict, visible_checking=visible_checking
        )

        edge_pcd = o3d.geometry.PointCloud()
        edge_pcd.points = o3d.utility.Vector3dVector(pred_edge_points)

        edge_ply_file_path = os.path.join(result_dir, "edge_points.ply")
        try:
            o3d.io.write_point_cloud(edge_ply_file_path, edge_pcd, write_ascii=True)
            logging.info(f"Saved {edge_ply_file_path} for edge points visualization.")
        except IOError as e:
            logging.error(f"Failed to save {edge_ply_file_path}: {e}")

        json_file_path = os.path.join(result_dir, "parametric_edges.json")
        try:
            with open(json_file_path, "w") as json_file:
                json.dump(return_edge_dict, json_file)
            logging.info(f"Saved {json_file_path} for evaluation.")
        except IOError as e:
            logging.error(f"Failed to save {json_file_path}: {e}")
