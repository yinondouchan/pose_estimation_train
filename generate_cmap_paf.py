import torch
import numpy as np

from enum import Enum


class AnnotationType(Enum):
    """ COCO keypoint annotation type """
    UNLABELED = 0
    LABELED_HIDDEN = 1
    LABELED_VISIBLE = 2

EPS = 1e-5


def generate_cmap(counts: torch.Tensor, peaks: torch.Tensor, height, width, stdev, window, device='cpu', kernel_type='gaussian'):
    C = peaks.size(0)  # number of part types
    M = peaks.size(1)  # max number of parts per type
    H = height
    W = width
    w = window / 2

    cmap = torch.zeros(C, H, W).to(device)
    var = stdev * stdev

    for c in range(C):

        count = int(counts[c])
        for p in range(count):
            i_mean = peaks[c][p][0] * H
            j_mean = peaks[c][p][1] * W
            i_min = int(i_mean - w)
            i_max = int(i_mean + w + 1)
            j_min = int(j_mean - w)
            j_max = int(j_mean + w + 1)
            
            if i_min < 0:
                i_min = 0
            if i_max < 0:
                i_max = 0
            if i_min >= H:
                i_min = H
            if i_max >= H:
                i_max = H
                
            if j_min < 0:
                j_min = 0
            if j_max < 0:
                j_max = 0
            if j_min >= W:
                j_min = w
            if j_max >= W:
                j_max = W

            # construct gaussian kernel
            d_i, d_j = torch.meshgrid(i_mean - (torch.arange(i_min, i_max) + 0.5),
                                      j_mean - (torch.arange(j_min, j_max) + 0.5))
            d_i = d_i.to(device)
            d_j = d_j.to(device)
            
            if kernel_type == 'gaussian':
                kernel = torch.exp((-d_i * d_i - d_j * d_j) / var)
            elif kernel_type == 'linear':
                kernel = (stdev - torch.abs(d_i) - torch.abs(d_j)) / stdev
                kernel = torch.max(kernel, torch.zeros_like(kernel))
            else:
                raise TypeError('Unknown kernel type: %s' % kernel_type)

            # add kernel to confidence map while preserving pixels of values higher than the assigned kernel
            cmap[c][i_min: i_max, j_min: j_max] = torch.max(cmap[c][i_min: i_max, j_min: j_max], kernel)

    return cmap
    
    
def generate_cmap_pinpoint(counts: torch.Tensor, peaks: torch.Tensor, height, width, amplify_output=False):
    C = peaks.size(0)  # number of part types
    M = peaks.size(1)  # max number of parts per type
    H = height
    W = width

    cmap = torch.zeros(C, H, W)

    for c in range(C):

        count = int(counts[c])
        for p in range(count):
            i_pos = float(peaks[c][p][0] * H)
            j_pos = float(peaks[c][p][1] * W)

            # find point's central location
            i_center = int(np.round(i_pos))
            j_center = int(np.round(j_pos))

            i_pos_rel = i_pos - i_center
            j_pos_rel = j_pos - j_center

            # add x and y components
            left_x = 0.5 - j_pos_rel
            right_x = 0.5 + j_pos_rel
            top_y = 0.5 - i_pos_rel
            bottom_y = 0.5 + i_pos_rel

            max_val = 0

            if check_bounds(j_center - 1, i_center - 1, H, W):
                cmap[c, i_center - 1, j_center - 1] = top_y * left_x
                max_val = max(max_val, top_y * left_x)
            if check_bounds(j_center - 1, i_center, H, W):
                cmap[c, i_center, j_center - 1] = bottom_y * left_x
                max_val = max(max_val, bottom_y * left_x)
            if check_bounds(j_center, i_center, H, W):
                cmap[c, i_center, j_center] = bottom_y * right_x
                max_val = max(max_val, bottom_y * right_x)
            if check_bounds(j_center, i_center - 1, H, W):
                cmap[c, i_center - 1, j_center] = top_y * right_x
                max_val = max(max_val, top_y * right_x)

            if amplify_output:
                # make max value to be 1
                if check_bounds(j_center - 1, i_center - 1, H, W):
                    cmap[c, i_center - 1, j_center - 1] /= max_val if max_val != 0 else 0
                if check_bounds(j_center - 1, i_center, H, W):
                    cmap[c, i_center, j_center - 1] /= max_val if max_val != 0 else 0
                if check_bounds(j_center, i_center, H, W):
                    cmap[c, i_center, j_center] /= max_val if max_val != 0 else 0
                if check_bounds(j_center, i_center - 1, H, W):
                    cmap[c, i_center - 1, j_center] /= max_val if max_val != 0 else 0

    return cmap


def check_bounds(x, y, H, W):
    return x < W and x >= 0 and y < H and y >= 0


def generate_paf(connections: torch.Tensor, topology: torch.Tensor, counts: torch.Tensor, peaks: torch.Tensor, height, width, stdev, window=None, device='cpu'):
    K = topology.size(0)
    H = height
    W = width

    paf = torch.zeros((2 * K, H, W)).to(device)
    var = stdev * stdev
    
    p_c_i, p_c_j = torch.meshgrid(torch.arange(0, H) + 0.5, torch.arange(0, W) + 0.5)
    p_c_i = p_c_i.to(device)
    p_c_j = p_c_j.to(device)

    for k in range(K):
        k_i = int(topology[k, 0])
        k_j = int(topology[k, 1])
        c_a = int(topology[k, 2])
        c_b = int(topology[k, 3])
        count = int(counts[c_a])

        for i_a in range(count):
            i_b = int(connections[k, 0, i_a])
            if i_b < 0:
                # connection doesn't exist
                continue

            p_a = peaks[c_a, i_a]
            p_b = peaks[c_b, i_b]

            p_a_i = p_a[0] * H
            p_a_j = p_a[1] * W
            p_b_i = p_b[0] * H
            p_b_j = p_b[1] * W
            p_ab_i = p_b_i - p_a_i
            p_ab_j = p_b_j - p_a_j
            p_ab_mag = float(torch.sqrt(p_ab_i * p_ab_i + p_ab_j * p_ab_j)) + EPS
            u_ab_i = p_ab_i / p_ab_mag
            u_ab_j = p_ab_j / p_ab_mag

            p_ac_i = p_c_i - p_a_i
            p_ac_j = p_c_j - p_a_j

            # dot product to find tangent bounds
            dot = p_ac_i * u_ab_i + p_ac_j * u_ab_j

            tandist = torch.zeros_like(p_c_i).to(device)
            tandist[dot < 0.0] = dot[dot < 0.0]
            tandist[dot > p_ab_mag] = (dot - p_ab_mag)[dot > p_ab_mag]

            # cross product to find perpendicular bounds
            cross = p_ac_i * u_ab_j - p_ac_j * u_ab_i

            # scale exponentially RBF by 2D distance from nearest point on line segment
            scale = torch.exp(-(tandist*tandist + cross*cross) / var)

            if window is not None:
                scale[cross > window] = 0
                scale[cross < -window] = 0

            paf[k_i] += scale * u_ab_i
            paf[k_j] += scale * u_ab_j

    return paf


def annotations_to_peaks(annotations_tensor: torch.Tensor, num_part_types, max_parts=100):
    """
    :param annotations_tensor: list of COCO keypoint annotations where each element is a list/tensor of one object's annotations
    :param num_part_types: number of part types
    :param max_parts: maximum number of parts per part type
    :return: a list of (link_type, x1, y1, x2, y2) tuples where link_type is the link type and x1, y1, x2, y2
    are the normalized coordinates of the link's points
    """
    peaks = torch.zeros(num_part_types, max_parts, 2)

    # maps each keypoint to its peak index in peaks tensor
    peak_inds = torch.zeros(annotations_tensor.shape[:-1]) - 1

    counts = torch.zeros(num_part_types)

    for i, annotations in enumerate(annotations_tensor):
        for j, annotation in enumerate(annotations):
            if annotation[2] == 0:
                # unlabeled keypoint - move on
                continue

            x = float(annotation[0])
            y = float(annotation[1])

            peaks[j, int(counts[j]), 0] = y
            peaks[j, int(counts[j]), 1] = x

            peak_inds[i][j] = counts[j]

            counts[j] += 1

    return peaks, counts, peak_inds


def annotations_to_connections(annotations_tensor: torch.Tensor, peak_inds: torch.Tensor, source_to_sink_map, link_to_connection_map, max_parts=100):
    connections = torch.zeros(len(link_to_connection_map), 2, max_parts) - 1

    for i, annotations in enumerate(annotations_tensor):
        for j, annotation in enumerate(annotations):
            if annotation[2] == 0:
                # unlabeled source keypoint - move on
                continue

            source_peak_ind = int(peak_inds[i][j])

            sink_part_inds = source_to_sink_map[j]
            if not type(sink_part_inds) is list:
                sink_part_inds = [sink_part_inds]

            for sink_part_ind in sink_part_inds:
                sink_peak_ind = int(peak_inds[i][sink_part_ind])

                if sink_peak_ind == -1:
                    # no sink keypoint of this type - move on
                    continue

                link_ind = link_to_connection_map[(j, sink_part_ind)]

                connections[link_ind, 0, source_peak_ind] = sink_peak_ind
                connections[link_ind, 1, sink_peak_ind] = source_peak_ind

    return connections


def annotations_to_link_data(annotations: torch.Tensor, source_to_sink_map: list, link_to_connection_map: dict, image_height, image_width):
    """
    :param annotations: list of COCO keypoint annotations
    :param source_to_sink_map: maps source part type to sink part type
    :param link_to_connection_map: maps source and sink part type tuples to connection index
    :param image_height: input image height
    :param image_width: input image width
    :return: a list of (link_type, x1, y1, x2, y2) tuples where link_type is the link type and x1, y1, x2, y2
    are the normalized coordinates of the link's points
    """

    link_data = list()

    # assume at most one instance per link type
    for i, annotation in enumerate(annotations):
        x1, y1, annotation_type = annotation.clone()
        if annotation_type == AnnotationType.UNLABELED:
            # source part is unlabeled
            continue

        # find sink part index
        sink_ind = source_to_sink_map[i]
        x2, y2, sink_annotation_type = annotations[sink_ind].clone()

        if sink_annotation_type == AnnotationType.UNLABELED:
            # source part exists but has no sink part connected to it
            continue

        # normalize source and sink coordinates
        x1 /= image_width
        y1 /= image_height
        x2 /= image_width
        y2 /= image_width
        link_type = link_to_connection_map[(i, sink_ind)]

        link_data.append((link_type, x1, y1, x2, y2))

    return link_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_parts = 6
    num_links = 6

    # topology tensor (source_paf_ind, sink_paf_ind, source_part_ind, sink_part_ind)
    topology = torch.Tensor([[0, 1, 0, 1],      # left big toe-> left small toe
                             [2, 3, 1, 2],      # left small toe -> heel
                             [4, 5, 2, 0],      # heel -> left big toe
                             [6, 7, 3, 4],      # right big toe-> right small toe
                             [8, 9, 4, 5],      # right small toe -> right heel
                             [10, 11, 5, 3]])     # right heel -> right big toe

    source_to_sink_map = {0: 1, 1: 2, 2: 0, 3: 4, 4: 5, 5: 3}

    # maps a link type to its index in the connections tensor
    link_to_connection_ind = {
        (0, 1): 0,
        (1, 2): 1,
        (2, 0): 2,
        (3, 4): 3,
        (4, 5): 4,
        (5, 3): 5
    }

    output_height = 56
    output_width = 56

    stdev = 1.0
    window = 5 * stdev

    keypoints_list = [[0, 0, 0, 0.5, 0.5, 2, 0.25, 0.25, 2, 0.1, 0.1, 2, 0.8, 0.8, 2, 0, 0, 0],
                      [0.71, 0.1, 2, 0.5, 0.2, 2, 0.45, 0.35, 2, 0.05, 0.7, 2, 0.4, 0.3, 2, 0, 0, 0]]
    annotations_list = torch.Tensor(keypoints_list).reshape(-1, int(len(keypoints_list[0]) / 3), 3)

    peaks, counts, peak_inds = annotations_to_peaks(annotations_list, num_parts)
    cmap = generate_cmap(counts, peaks, 56, 56, stdev, window)
    connections = annotations_to_connections(annotations_list, peak_inds, source_to_sink_map, link_to_connection_ind)
    paf = generate_paf(connections, topology, counts, peaks, 56, 56, stdev)

    plt.imshow(paf[4])
    plt.show()
    pass