import torch
from torch.nn import functional as F


def get_udf_normals_grid(
    func,
    func_grad,
    N,
    udf_threshold,
    is_linedirection=False,
    sampling_N=50,
    sampling_delta=0.005,
    max_batch=int(2**12),
    device="cuda",
):
    """
    Efficiently fills a dense N*N*N regular grid by querying the function and its gradient for distance field values
    and optionally computing line directions. Adjusts voxel grid based on specified origin and max values.

    Parameters:
    - func: Callable for evaluating distance field values.
    - func_grad: Callable for evaluating gradients.
    - N: Size of the grid in each dimension.
    - udf_threshold: Threshold below which gradients are computed.
    - is_linedirection: Flag indicating whether to compute line directions.
    - sampling_N: Number of samples for line direction computation.
    - sampling_delta: Offset range for sampling around points for line direction.
    - max_batch: Max number of points processed in a single batch.

    Returns:
    Tuple of tensors (df_values, line_directions, gradients, samples, voxel_size) representing the computed
    distance field values, line directions, gradients at points below threshold, raw sample points, and the size
    of each voxel.
    """

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 12, device=device)
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = torch.div(overall_index, N, rounding_mode="floor") % N
    samples[:, 0] = (
        torch.div(
            torch.div(overall_index, N, rounding_mode="floor"), N, rounding_mode="floor"
        )
        % N
    )
    # Ensure voxel_origin and voxel_max are correctly set
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    # Query function for distance field values
    num_samples = N**3
    samples.requires_grad = False
    for head in range(0, num_samples, max_batch):
        tail = min(head + max_batch, num_samples)
        sample_subset = samples[head:tail, :3].clone().to(device)
        df, _, _ = func(sample_subset)
        samples[head:tail, 3:4] = df.detach()

    # Compute gradients where distance field value is below threshold
    norm_mask = samples[:, 3] < udf_threshold
    norm_idx = torch.where(norm_mask)[0]
    for head in range(0, len(norm_idx), max_batch):
        tail = min(head + max_batch, len(norm_idx))
        idx_subset = norm_idx[head:tail]
        sample_subset = samples[idx_subset, :3].clone().to(device).requires_grad_(True)
        grad = func_grad(sample_subset).detach()
        samples[idx_subset, 4:7] = -F.normalize(grad, dim=1)[:, 0]

        # Compute line directions if requested
        if is_linedirection:
            sample_subset_ld = sample_subset.unsqueeze(
                1
            ) + sampling_delta * torch.randn(
                (sample_subset.shape[0], sampling_N, 3), device=device
            )
            grad_ld = (
                func_grad(sample_subset_ld.reshape(-1, 3))
                .detach()
                .reshape(sample_subset.shape[0], sampling_N, 3)
            )
            _, _, vh = torch.linalg.svd(grad_ld)
            null_space = vh[:, -1, :].view(-1, 3)
            samples[idx_subset, 8:11] = F.normalize(null_space, dim=1)

    # Reshape output tensors
    df_values = samples[:, 3].reshape(N, N, N)
    vecs = samples[:, 4:7].reshape(N, N, N, 3)
    ld = samples[:, 8:11].reshape(N, N, N, 3)

    return df_values, ld, vecs, samples, torch.tensor(voxel_size)


def get_udf_normals_slow(
    func,
    func_grad,
    voxel_size,
    xyz,
    is_linedirection,
    # N_ld=20,
    sampling_N=50,
    sampling_delta=0.005,  # 0.005
    max_batch=int(2**12),
    device="cuda",
):
    """
    Computes distance field values, normals, and optionally line directions for a set of points.

    Parameters:
    - func: Function to evaluate the distance field.
    - func_grad: Function to evaluate the gradient of the distance field.
    - voxel_size: Size of the voxel (not used in this function but kept for compatibility).
    - xyz: (N,3) tensor representing coordinates to evaluate.
    - is_linedirection: Boolean indicating whether to compute line directions.
    - sampling_N: Number of samples for computing line direction.
    - sampling_delta: Delta range for sampling around points for line direction.
    - max_batch: Maximum number of points to process in a single batch.

    Returns:
    - df_values: (N,) tensor of distance field values at xyz locations.
    - normals: (N,3) tensor of gradient values at xyz locations.
    - ld: (N,3) tensor of line direction values at xyz locations, if computed.
    - samples: (N, 10) tensor of x, y, z, distance field, grad_x, grad_y, grad_z, and optionally line directions.
    """
    # network.eval()
    ################
    # transform first 3 columns
    # to be the x, y, z coordinate

    num_samples = xyz.shape[0]
    # xyz = torch.from_numpy(xyz).float().cuda()
    samples = torch.cat([xyz, torch.zeros(num_samples, 10).float().cuda()], dim=-1)
    samples.requires_grad = False
    # samples.pin_memory()
    ################
    # 2: Run forward pass to fill the grid
    ################
    head = 0
    ## FIRST: fill distance field grid without gradients
    while head < num_samples:
        # xyz coords
        sample_subset = (
            samples[head : min(head + max_batch, num_samples), 0:3].clone().cuda()
        )
        # Create input
        xyz = sample_subset

        input = xyz.reshape(-1, xyz.shape[-1])
        # Run forward pass
        with torch.no_grad():
            df, _, PE = func(input)
        # Store df
        samples[head : min(head + max_batch, num_samples), 3] = (
            df.squeeze(-1).detach().cpu()
        )
        grad = func_grad(input).detach()[:, 0]
        normals = -F.normalize(grad, dim=1)
        samples[head : min(head + max_batch, num_samples), 4:7] = normals.cpu()

        if is_linedirection:
            input_ld = input.unsqueeze(
                1
            ) + sampling_delta * torch.randn(  # need to be fixed grid
                (input.shape[0], sampling_N, 3), device=device
            )
            # input_ld = input.unsqueeze(1) + offset
            input_ld = input_ld.reshape(-1, input.shape[-1])
            grad_ld = (
                func_grad(input_ld.float())
                .detach()[:, 0]
                .reshape(input.shape[0], -1, 3)
            )
            _, _, vh = torch.linalg.svd(grad_ld)

            # Extract the null space (non-zero solutions) for each matrix in the batch
            # null_space = vh[:, -1, :][vh[:, -1, :] != 0].view(-1, 3)
            null_space = vh[:, -1, :].view(-1, 3)
            samples[head : min(head + max_batch, num_samples), 7:10] = F.normalize(
                null_space, dim=1
            )
        # Next iter
        head += max_batch

    # Separate values in DF / gradients
    df_values = samples[:, 3]
    normals = samples[:, 4:7]
    ld = samples[:, 7:10]

    return df_values, normals, ld, samples


import numpy as np


def project_vector_onto_plane(A, B):
    # Calculate the dot product of A and B
    dot_product = torch.sum(A * B, dim=-1)

    # Calculate the projection of A onto B
    projection = dot_product.unsqueeze(-1) * B

    # Calculate the projected vector onto the plane perpendicular to B
    projected_vector = A - projection

    return projected_vector


def get_pointcloud_from_udf(
    func,
    func_grad,
    N_MC=128,
    udf_threshold=1.0,
    sampling_N=50,
    sampling_delta=5e-3,
    is_pointshift=False,
    iters=1,
    is_linedirection=False,
    device="cuda",
):
    """
    Computes a point cloud from a distance field network conditioned on the latent vector.
    Inputs:
        func: Function to evaluate the distance field.
        func_grad: Function to evaluate the gradient of the distance field.
        N_MC: Size of the grid.
        udf_threshold: Threshold to filter surfaces with large UDF values.
        is_pointshift: Flag indicating if points should be shifted by df * normals.
        iters: Number of iterations for the point shift.
        is_linedirection: Flag indicating if line direction computation is needed.
    Returns:
        pointcloud: (N**3, 3) tensor representing the edge point cloud.
        samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z).
        indices: Indices of coordinates that need updating in the next iteration.
    """
    # Compute UDF normals and grid
    df_values, lds, normals, samples, voxel_size = get_udf_normals_grid(
        func=func,
        func_grad=func_grad,
        N=N_MC,
        udf_threshold=udf_threshold,
        is_linedirection=is_linedirection,
        sampling_N=sampling_N,
        sampling_delta=sampling_delta,
        device=device,
    )

    # Reshape tensors for processing
    df_values, lds, normals, samples = (
        df_values.reshape(-1),
        lds.reshape(-1, 3),
        normals.reshape(-1, 3),
        samples.reshape(-1, 12),
    )
    xyz = samples[:, 0:3]
    df_values.clamp_(min=0)  # Ensure distance field values are non-negative

    # Filter out points too far from the surface
    points_idx = df_values <= udf_threshold
    filtered_xyz, filtered_lds, normals, df_values = (
        xyz[points_idx],
        lds[points_idx],
        normals[points_idx],
        df_values[points_idx],
    )

    # Point shifting
    if is_pointshift and iters > 0:
        for iter in range(iters):
            shifted_xyz = filtered_xyz + df_values.unsqueeze(-1) * normals
            shifted_df_values, shifted_normals, filtered_lds, _ = get_udf_normals_slow(
                func=func,
                func_grad=func_grad,
                voxel_size=voxel_size,
                xyz=shifted_xyz,
                is_linedirection=True if iter == iters - 1 else False,
                device=device,
            )
            shifted_points_idx = shifted_df_values <= udf_threshold
            filtered_xyz, df_values, normals, filtered_lds = (
                shifted_xyz[shifted_points_idx],
                shifted_df_values[shifted_points_idx],
                shifted_normals[shifted_points_idx],
                filtered_lds[shifted_points_idx],
            )

    return (
        filtered_xyz.cpu().numpy(),
        filtered_lds.cpu().numpy() if filtered_lds is not None else None,
    )
