import os
import pytest
import numpy as np
import pandas as pd
from full_crack_analysis import through_chart, edge_stacked_chart, phases_cracks_to_csv, skeleton_neighbors_to_csv


def test_through_chart(tmp_path):
    phase_through_ids = np.array([-1, 0, 1, 2])
    phase_through_count = np.array([0, 10, 20, 5])
    labels = np.array(["Phase0", "Phase1", "Phase2"])
    
    # Global output_dir_path is used in through_chart or can be monkeypatched
    import full_crack_analysis
    full_crack_analysis.output_dir_path = str(tmp_path)
    
    through_chart(phase_through_ids, phase_through_count, labels, "sample_test", "test_")
    assert os.path.exists(os.path.join(tmp_path, "test_phase-through.png"))


def test_edge_stacked_chart(tmp_path):
    import full_crack_analysis
    full_crack_analysis.output_dir_path = str(tmp_path)

    phase_edge_pairs = np.array([
        [-1, 0],
        [0, 1],
        [1, 2]
    ])
    phase_edge_count = np.array([5, 15, 25])
    labels = ["Phase0", "Phase1", "Phase2"]

    all_phases_used, edge_counts = edge_stacked_chart(
        phase_edge_pairs, phase_edge_count, labels, "sample_test", "test_"
    )
    assert os.path.exists(os.path.join(tmp_path, "test_phase-edge.png"))
    assert len(all_phases_used) > 0
    assert edge_counts.shape == (len(all_phases_used), len(all_phases_used))


def test_phases_cracks_to_csv(tmp_path):
    labels = np.array(["Phase0", "Phase1", "Phase2"])
    all_phases_used = np.array([-1, 0, 1, 2])
    edge_counts = np.zeros((4, 4), dtype=int)
    edge_counts[1, 2] = 10
    edge_counts[2, 1] = 10
    phase_through_ids = np.array([-1, 0, 1])
    phase_through_count = np.array([0, 5, 8])

    phases_cracks_to_csv(
        labels, edge_counts, phase_through_ids, phase_through_count,
        all_phases_used, str(tmp_path), "test_"
    )
    csv_file = os.path.join(tmp_path, "test_phases_cracks.csv")
    assert os.path.exists(csv_file)
    df = pd.read_csv(csv_file)
    assert "phase id" in df.columns
    assert "label" in df.columns


def test_skeleton_neighbors_to_csv(tmp_path):
    shape = (20, 20)
    gradient_map = np.ones(shape + (2,), dtype=float)
    neigh_coords = np.zeros(shape + (2, 2), dtype=int)
    # populate one skeleton pixel at (10, 10)
    neigh_coords[10, 10, 0] = [10, 9]   # left neighbor
    neigh_coords[10, 10, 1] = [10, 11]  # right neighbor
    # rest are invalid -1
    neigh_coords[:10, :, :, :] = -1
    neigh_coords[11:, :, :, :] = -1
    neigh_coords[10, :10, :, :] = -1
    neigh_coords[10, 11:, :, :] = -1

    smooth_mask = np.zeros(shape, dtype=np.uint8)
    smooth_mask[10, 10] = 1

    branches_map = np.zeros(shape, dtype=int)
    branches_map[10, 10] = 1

    phase_map_img = np.ones(shape, dtype=int)
    phase_map_img[10, 9] = 1
    phase_map_img[10, 11] = 2

    skeleton_neighbors_to_csv(
        gradient_map, neigh_coords, smooth_mask, branches_map, phase_map_img,
        str(tmp_path), "test_"
    )
    csv_file = os.path.join(tmp_path, "test_skeleton-neighbors.csv")
    assert os.path.exists(csv_file)
    df = pd.read_csv(csv_file)
    assert len(df) == 1
    assert df["crack skeleton coord X"].iloc[0] == 10
    assert df["crack skeleton coord Y"].iloc[0] == 10
