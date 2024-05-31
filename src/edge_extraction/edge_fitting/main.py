import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d
import random
from src.edge_extraction.edge_fitting.bezier_fit import bezier_fit, bezier_curve
from src.edge_extraction.edge_fitting.line_fit import fit_line_ransac_3d


class LineSegment:
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point


def generate_segments_from_idx(connected_lines, points_wld):
    segments = []
    polylines_wld = []
    for connected_line in connected_lines:
        polyline_wld = [points_wld[connected_line[0]].tolist()]
        for i in range(len(connected_line) - 1):
            segment = [
                points_wld[connected_line[i]].tolist(),
                points_wld[connected_line[i + 1]].tolist(),
            ]
            segments.append(segment)
            polyline_wld.append(points_wld[connected_line[i + 1]])
        polyline_wld = np.array(polyline_wld).reshape(-1, 6)
        polylines_wld.append(polyline_wld)

    return np.array(segments).reshape(-1, 6), polylines_wld


def create_line_segments_from_3d_array(segment_data):
    x1, y1, z1, x2, y2, z2 = (
        segment_data[:, 0],
        segment_data[:, 1],
        segment_data[:, 2],
        segment_data[:, 3],
        segment_data[:, 4],
        segment_data[:, 5],
    )
    segments = []
    for i in range(len(x1)):
        segments.append(
            LineSegment(
                np.array([x1[i], y1[i], z1[i]]), np.array([x2[i], y2[i], z2[i]])
            )
        )
    return segments


def is_point_inside_ranges(point, ranges):
    point = np.array(point)
    if not np.all(point > ranges[0]) or not np.all(point < ranges[1]):
        return False
    return True


def is_line_inside_ranges(line_segment, ranges):
    if not is_point_inside_ranges(line_segment.start_point, ranges):
        return False
    if not is_point_inside_ranges(line_segment.end_point, ranges):
        return False
    return True


def create_open3d_line_set(
    line_segments, color=[0.5, 0.5, 0.5], width=2, ranges=None, scale=1.0
):
    o3d_points, o3d_lines, o3d_colors = [], [], []
    counter = 0
    for line_segment in line_segments:
        if ranges is not None:
            if not is_line_inside_ranges(line_segment, ranges):
                continue
        o3d_points.append(line_segment.start_point * scale)
        o3d_points.append(line_segment.end_point * scale)
        o3d_lines.append([2 * counter, 2 * counter + 1])
        counter += 1
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(o3d_points)
    line_set.lines = o3d.utility.Vector2iVector(o3d_lines)
    line_set.colors = o3d.utility.Vector3dVector(o3d_colors)
    return line_set


def save_3d_lines_to_file(line_segments, filename, width=2, ranges=None, scale=1.0):
    lines = create_line_segments_from_3d_array(line_segments)
    line_set = create_open3d_line_set(lines, width=width, ranges=ranges, scale=scale)
    o3d.io.write_line_set(filename, line_set)


def connect_points(
    points, distance_threshold, angle_threshold, nms_factor, keep_short_lines
):
    num_points = len(points)
    connected_line_segments = []

    unvisited_points = set(range(num_points))

    while len(unvisited_points) > 0:
        selected_point = np.random.choice(list(unvisited_points))
        selected_point_opposite = selected_point

        unvisited_points.remove(selected_point)
        connected_line = [selected_point]

        while True:  # forward connection
            dist = cdist(
                [points[selected_point, :3]], points[list(unvisited_points), :3]
            )
            neighboring_points = np.where(dist < distance_threshold)[1]
            neighboring_distance = dist[0, neighboring_points].reshape(-1)

            neighboring_points = (
                np.array(list(unvisited_points))[neighboring_points]
            ).tolist()

            if len(neighboring_points) == 0:
                break

            directions = (
                points[neighboring_points, :3] - points[selected_point, :3][None, ...]
            )
            directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis] + 1e-6

            dot_products = np.dot(directions, points[selected_point, 3:])

            closest_point_idx = np.argmax(dot_products)

            if (
                dot_products[closest_point_idx] <= 1 - angle_threshold
            ):  # no suitable point found
                break

            connected_line.append(
                neighboring_points[closest_point_idx]
            )  # add the point to the line

            invalid_points_idx = np.where(
                (neighboring_distance <= neighboring_distance[closest_point_idx])
                * (dot_products < dot_products[closest_point_idx])
                * (dot_products >= nms_factor * dot_products[closest_point_idx])
            )[0]
            invalid_points = np.array(neighboring_points)[invalid_points_idx].tolist()

            unvisited_points.difference_update(invalid_points)

            if (
                np.dot(
                    points[neighboring_points[closest_point_idx], 3:],
                    directions[closest_point_idx],
                )
                <= 0.5
            ):  #
                break

            unvisited_points.remove(
                neighboring_points[closest_point_idx]
            )  # remove connected point from unvisited points set
            selected_point = neighboring_points[
                closest_point_idx
            ]  # update anchor point

        while True:  # backward connection
            dist = cdist(
                [points[selected_point_opposite, :3]],
                points[list(unvisited_points), :3],
            )
            neighboring_points = np.where(dist < distance_threshold)[1]
            neighboring_distance = dist[0, neighboring_points].reshape(-1)

            neighboring_points = (
                np.array(list(unvisited_points))[neighboring_points]
            ).tolist()

            if len(neighboring_points) == 0:
                break

            directions = (
                points[neighboring_points, :3]
                - points[selected_point_opposite, :3][None, ...]
            )
            directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis] + 1e-6

            dot_products = np.dot(directions, points[selected_point_opposite, 3:])

            closest_point_idx = np.argmin(dot_products)

            if (
                abs(dot_products[closest_point_idx]) <= 1 - angle_threshold
                or dot_products[closest_point_idx] >= 0
            ):
                break

            connected_line.insert(
                0, neighboring_points[closest_point_idx]
            )  # add connected point to the beginning of the line

            invalid_points_idx = np.where(
                (neighboring_distance <= neighboring_distance[closest_point_idx])
                * (dot_products > dot_products[closest_point_idx])
                * (dot_products <= nms_factor * dot_products[closest_point_idx])
            )[0]
            invalid_points = np.array(neighboring_points)[invalid_points_idx].tolist()

            unvisited_points.difference_update(invalid_points)

            if (
                np.dot(
                    -points[neighboring_points[closest_point_idx], 3:],
                    directions[closest_point_idx],
                )
                <= 0.5
            ):
                break

            unvisited_points.remove(neighboring_points[closest_point_idx])
            selected_point_opposite = neighboring_points[closest_point_idx]

        if not keep_short_lines:
            if len(connected_line) > 3:
                connected_line_segments.append(connected_line)
        else:
            if len(connected_line) > 1:
                connected_line_segments.append(connected_line)

    return connected_line_segments


def edge_fitting(
    polylines_wld,
    voxel_size=256,
    max_iterations=100,
    min_inliers=4,
    max_lines=3,
    max_curves=2,
    keep_short_lines=True,
):
    straight_lines = []
    raw_points_on_straight_lines = []
    bezier_curve_params = []
    bezier_curve_points = []
    raw_points_on_curves = []
    t_fit = np.linspace(0, 1, 100)
    for endpoints_wld in polylines_wld:
        if len(endpoints_wld) < 4 and keep_short_lines:  # keep short lines

            for i in range(len(endpoints_wld) - 1):
                segment = [
                    endpoints_wld[i, :3],
                    endpoints_wld[i + 1, :3],
                ]
                straight_lines.append(np.array(segment).reshape(-1))
                raw_points_on_straight_lines.extend(
                    [np.array(segment).reshape(-1, 3).tolist()]
                )
        else:
            (
                straight_line,
                split_points,
                potential_curve_points,
            ) = fit_line_ransac_3d(
                endpoints_wld,
                voxel_size,
                max_iterations,
                min_inliers,
                max_lines,
                max_curves,
                keep_short_lines,
            )

            if len(split_points) >= 1:  # fitted straight lines exist
                straight_lines.extend(straight_line)
                raw_points_on_straight_lines.extend(split_points)

            if len(potential_curve_points) >= 1:  # fitted curves exist
                for curve_points in potential_curve_points:
                    p = bezier_fit(curve_points, error_threshold=5.0 / voxel_size)
                    if p is None:
                        print("Fitting error too high, not fitting")
                        continue
                    bezier_curve_params.append(p)

                    xyz_fit = bezier_curve(t_fit, *p).reshape(-1, 3)
                    bezier_curve_points.append(xyz_fit)
                    raw_points_on_curves.append(curve_points.tolist())

    straight_lines = np.array(straight_lines)

    if len(bezier_curve_points) >= 1:
        bezier_curve_points = np.concatenate(bezier_curve_points, axis=0)
        bezier_curve_params = np.array(bezier_curve_params)

    return (
        straight_lines,
        raw_points_on_straight_lines,
        bezier_curve_params,
        bezier_curve_points,
        raw_points_on_curves,
    )


def edge_fit(
    edge_data=None,
    angle_threshold=0.03,
    nms_factor=0.9,
    fit_distance_threshold=10.0,
    min_inliers=4,
    max_lines=4,
    max_curves=3,
    keep_short_lines=True,
):
    res = np.array(edge_data["resolution"])
    raw_points = np.array(edge_data["points"])
    raw_ld_colors = np.array(edge_data["ld_colors"])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_points)
    pcd.colors = o3d.utility.Vector3dVector(raw_ld_colors)
    fit_distance_threshold = fit_distance_threshold / res
    pcd = pcd.voxel_down_sample(voxel_size=2.0 / res)

    points = np.asarray(pcd.points)
    ld_colors = np.asarray(pcd.colors)
    linedirection = ld_colors * 2 - 1  # recover the line direction
    linedirection = linedirection / (
        np.linalg.norm(linedirection, axis=1)[:, np.newaxis] + 1e-6
    )
    points_wld = np.concatenate((points, linedirection), axis=1)

    connected_line = connect_points(
        points_wld,
        fit_distance_threshold,
        angle_threshold,
        nms_factor,
        keep_short_lines,
    )

    _, polylines_wld = generate_segments_from_idx(connected_line, points_wld)

    (
        straight_lines,
        raw_points_on_straight_lines,
        bezier_curve_params,
        bezier_curve_points,
        raw_points_on_curves,
    ) = edge_fitting(
        polylines_wld,
        voxel_size=res,
        max_iterations=100,
        min_inliers=min_inliers,
        max_lines=max_lines,
        max_curves=max_curves,
        keep_short_lines=keep_short_lines,
    )

    fitted_edge_dict = {
        "resolution": int(res),
        "lines_end_pts": straight_lines.tolist() if len(straight_lines) > 0 else [],
        "raw_points_on_lines": (
            raw_points_on_straight_lines
            if len(raw_points_on_straight_lines) > 0
            else []
        ),
        "curves_ctl_pts": (
            bezier_curve_params.tolist() if len(bezier_curve_params) > 0 else []
        ),
        "raw_points_on_curves": (
            raw_points_on_curves if len(raw_points_on_curves) > 0 else []
        ),
    }

    return fitted_edge_dict
