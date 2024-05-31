import torch
import torch.nn as nn
import numpy as np
from src.models.embedder import get_embedder


class UDFNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=0,
        scale=1,
        bias=0.5,
        geometric_init=True,
        weight_norm=True,
        udf_type="abs",
    ):
        super(UDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        self.geometric_init = geometric_init

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    # torch.nn.init.constant_(lin.bias, bias)    # for indoor sdf setting
                    torch.nn.init.constant_(lin.bias, -bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )

            if weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.udf_type = udf_type

    def udf_out(self, x):
        if self.udf_type == "abs":
            return torch.abs(x)
        elif self.udf_type == "square":
            return x**2
        elif self.udf_type == "sdf":
            return x

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return (
            torch.cat([self.udf_out(x[:, :1]) / self.scale, x[:, 1:]], dim=-1),
            inputs,
        )

    def udf(self, x):
        feature_out, PE = self.forward(x)
        udf = feature_out[:, :1]
        feature = feature_out[:, 1:]
        return udf, feature, PE

    def udf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.set_grad_enabled(True):
            y = self.udf(x)[0]

            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(
        self,
        d_feature,
        mode,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        weight_norm=True,
        multires_view=0,
        squeeze_out=True,
    ):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        self.d_out = d_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0 and self.mode != "no_view_dir":
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 3

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None
        normals = normals.detach()
        if self.mode == "idr":
            rendering_input = torch.cat(
                [points, view_dirs, normals, -1 * normals, feature_vectors], dim=-1
            )
        elif self.mode == "no_view_dir":
            rendering_input = torch.cat(
                [points, normals, -1 * normals, feature_vectors], dim=-1
            )
        elif self.mode == "no_normal":
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            color = torch.sigmoid(x[:, : self.d_out])
        else:
            color = x[:, : self.d_out]

        return color


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, requires_grad=True):
        super(SingleVarianceNetwork, self).__init__()
        self.variance = nn.Parameter(
            torch.Tensor([init_val]), requires_grad=requires_grad
        )
        self.second_variance = nn.Parameter(
            torch.Tensor([init_val]), requires_grad=requires_grad
        )

    def set_trainable(self):
        self.variance.requires_grad = True
        self.second_variance.requires_grad = True

    def forward(self, x):
        return torch.ones([len(x), 1]).to(x.device) * torch.exp(self.variance * 10.0)

    def get_secondvariance(self, x):
        return torch.ones([len(x), 1]).to(x.device) * torch.exp(
            self.second_variance * 10.0
        )


class BetaNetwork(nn.Module):
    def __init__(
        self,
        init_var_beta=0.1,
        init_var_gamma=0.1,
        init_var_zeta=0.05,
        beta_min=0.00005,
        requires_grad_beta=True,
        requires_grad_gamma=True,
        requires_grad_zeta=True,
    ):
        super().__init__()

        self.beta = nn.Parameter(
            torch.Tensor([init_var_beta]), requires_grad=requires_grad_beta
        )
        self.gamma = nn.Parameter(
            torch.Tensor([init_var_gamma]), requires_grad=requires_grad_gamma
        )
        self.zeta = nn.Parameter(
            torch.Tensor([init_var_zeta]), requires_grad=requires_grad_zeta
        )
        self.beta_min = beta_min

    def get_beta(self):
        return torch.exp(self.beta * 10).clip(0, 1.0 / self.beta_min)

    def get_gamma(self):
        return torch.exp(self.gamma * 10)

    def get_zeta(self):
        """
        used for udf2prob mapping zeta*x/(1+zeta*x)
        :return:
        :rtype:
        """
        return self.zeta.abs()

    def set_beta_trainable(self):
        self.beta.requires_grad = True

    @torch.no_grad()
    def set_gamma(self, x):
        self.gamma = nn.Parameter(
            torch.Tensor([x]), requires_grad=self.gamma.requires_grad
        ).to(self.gamma.device)

    def forward(self):
        beta = self.get_beta()
        gamma = self.get_gamma()
        zeta = self.get_zeta()
        return beta, gamma, zeta
