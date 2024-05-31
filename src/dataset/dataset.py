import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from pathlib import Path
import json
import random


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print("Load data: Begin")
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.conf = conf
        self.scan = conf.get_string("scan")
        self.data_dir = os.path.join(conf.get_string("data_dir"), self.scan)
        self.dataset_name = self.conf.get_string("dataset_name", default="ABC")
        self.detector = conf.get_string("detector", default="DexiNed")
        assert self.detector in ["DexiNed", "PidiNet"]
        self.load_metadata(conf)
        self.load_image_data()
        print("Load data: End")

    def load_metadata(self, conf):
        meta = load_from_json(Path(self.data_dir) / "meta_data.json")
        self.meta = meta
        self.intrinsics_all = []
        self.pose_all = []
        self.edges_list = []
        self.colors_list = []

        meta_scene_box = meta["scene_box"]

        self.near = meta_scene_box["near"]
        self.far = meta_scene_box["far"]
        self.radius = meta_scene_box["radius"]

        H, W = meta["height"], meta["width"]
        self.H, self.W, self.image_pixels = H, W, H * W

        for idx, frame in enumerate(meta["frames"]):
            self.process_frame(frame)

    def process_frame(self, frame):
        intrinsics = torch.tensor(frame["intrinsics"])
        camtoworld = torch.tensor(frame["camtoworld"])[:4, :4]
        image_name = frame["rgb_path"]
        if self.detector == "PidiNet":
            self.edges_list.append(
                os.path.join(
                    self.data_dir,
                    "edge_PidiNet",
                    image_name[:-4] + ".png",
                )
            )
        elif self.detector == "DexiNed":
            self.edges_list.append(
                os.path.join(self.data_dir, "edge_DexiNed", image_name)
            )
        self.colors_list.append(os.path.join(self.data_dir, "color", image_name))
        self.intrinsics_all.append(intrinsics)
        self.pose_all.append(camtoworld)

    def load_image_data(self):
        self.load_edges_data()
        self.colors_np = (
            np.stack([cv.imread(im_name) for im_name in self.colors_list]) / 255.0
        )

        self.n_images = len(self.edges_list)
        self.edges = torch.from_numpy(
            self.edges_np.astype(np.float32)
        )  # .to(self.device)
        self.colors = torch.from_numpy(self.colors_np.astype(np.float32))

        # .to(
        #     self.device
        # )

        self.masks_np = (self.edges_np > 0.5).astype(np.float32)
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32))

        self.intrinsics_all = torch.stack(self.intrinsics_all)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all)

        self.object_bbox_min = np.array(self.meta["scene_box"]["aabb"][0])
        self.object_bbox_max = np.array(self.meta["scene_box"]["aabb"][1])

    def load_edges_data(self):
        edges = [cv.imread(im_name, 0)[..., None] for im_name in self.edges_list]
        self.edges_np = np.stack(edges) / 255.0

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        )  # W, H, 3
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        depth_scale = rays_v[:, :, 2:]
        rays_v = torch.matmul(
            self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(
            rays_v.shape
        )  # W, H, 3
        pose = self.pose_all[img_idx]  # [4, 4]
        intrinsics = self.intrinsics_all[img_idx]  # [4, 4]
        return (
            rays_o.transpose(0, 1).to(self.device),
            rays_v.transpose(0, 1).to(self.device),
            pose.to(self.device),
            intrinsics.to(self.device),
            depth_scale.to(self.device),
        )

    def gen_one_ray_at(self, img_idx, x, y):
        """

        Parameters
        ----------
        img_idx :
        x : for width
        y : for height

        Returns
        -------

        """
        image = np.uint8(self.edges_np[img_idx] * 256)
        image2 = cv.circle(image, (x, y), radius=10, color=(0, 0, 255), thickness=-1)

        pixels_x = torch.Tensor([x]).long()
        pixels_y = torch.Tensor([y]).long()
        edge = self.edges[img_idx][(pixels_y, pixels_x)]  # batch_size, 1
        color = self.colors[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = (self.masks[img_idx][(pixels_y, pixels_x)] > 0).to(
            torch.float32
        )  # batch_size, 1

        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        ).float()  # batch_size, 3
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]
        ).squeeze(
            -1
        )  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(
            self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]
        ).squeeze(
            -1
        )  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(
            rays_v.shape
        )  # batch_size, 3

        return (
            {
                "rays_o": rays_o.to(self.device),
                "rays_v": rays_v.to(self.device),
                "edge": edge.to(self.device),
                "color": color.to(self.device),
                "mask": mask[:, :1].to(self.device),
            },
            image2,
        )

    def gen_random_rays_patches_at(
        self,
        img_idx,
        batch_size,
        importance_sample=False,
    ):
        """
        Generate random rays at world space from one camera.
        """

        if not importance_sample:
            pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        elif (
            importance_sample and self.masks is not None
        ):  # sample more pts in the valid mask regions
            img_np = self.edges[img_idx].cpu().numpy()
            edge_density = np.mean(img_np)
            probabilities = np.ones_like(img_np) * edge_density
            probabilities[img_np > 0.1] = 1.0 - edge_density
            probabilities = probabilities.reshape(-1)

            # randomly sample 50%
            pixels_x_1 = torch.randint(low=0, high=self.W, size=[batch_size // 2])
            pixels_y_1 = torch.randint(low=0, high=self.H, size=[batch_size // 2])

            ys, xs = torch.meshgrid(
                torch.linspace(0, self.H - 1, self.H),
                torch.linspace(0, self.W - 1, self.W),
            )  # pytorch's meshgrid has indexing='ij'
            p = torch.stack([xs, ys], dim=-1)  # H, W, 2
            p_valid = p[self.masks[img_idx][:, :, 0] >= 0]  # [num, 2]

            # randomly sample 50% mainly from edge regions
            number_list = np.arange(self.image_pixels)
            random_idx = random.choices(number_list, probabilities, k=batch_size // 2)
            random_idx = torch.from_numpy(np.array(random_idx)).to(
                torch.int64
            )  # .to(self.device)
            p_select = p_valid[random_idx]  # [N_rays//2, 2]
            pixels_x_2 = p_select[:, 0]
            pixels_y_2 = p_select[:, 1]

            pixels_x = torch.cat([pixels_x_1, pixels_x_2], dim=0).to(torch.int64)
            pixels_y = torch.cat([pixels_y_1, pixels_y_2], dim=0).to(torch.int64)
        # normalized ndc uv coordinates, (-1, 1)
        ndc_u = 2 * pixels_x / (self.W - 1) - 1
        ndc_v = 2 * pixels_y / (self.H - 1) - 1
        rays_ndc_uv = torch.stack([ndc_u, ndc_v], dim=-1).view(-1, 2).float()

        edge = self.edges[img_idx][(pixels_y, pixels_x)]  # batch_size, 1
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        ).float()  # batch_size, 3
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]
        ).squeeze()  # batch_size, 3

        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        depth_scale = rays_v[:, 2:]  # batch_size, 1
        rays_v = torch.matmul(
            self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]
        ).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(
            rays_v.shape
        )  # batch_size, 3

        rays = {
            "rays_o": rays_o.to(self.device),
            "rays_v": rays_v.to(self.device),
            "edge": edge.to(self.device),
        }

        pose = self.pose_all[img_idx]  # [4, 4]
        intrinsics = self.intrinsics_all[img_idx]  # [4, 4]

        sample = {
            "rays": rays,
            "pose": pose,
            "intrinsics": intrinsics,
            "rays_ndc_uv": rays_ndc_uv.to(self.device),
            "rays_norm_XYZ_cam": p.to(self.device),  # - XYZ_cam, before multiply depth,
            "depth_scale": depth_scale.to(self.device),
        }

        return sample

    def edge_at(self, idx, resolution_level):
        edge = cv.imread(self.edges_list[idx], 0)[..., None]
        edge = (
            cv.resize(edge, (self.W // resolution_level, self.H // resolution_level))
        ).clip(0, 255)
        return edge

    def color_at(self, idx, resolution_level):
        img = cv.imread(self.colors_list[idx])
        img = cv.resize(
            img,
            (self.W // resolution_level, self.H // resolution_level),
            interpolation=cv.INTER_NEAREST,
        )
        return img
