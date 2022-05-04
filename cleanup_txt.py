"""
File to clean up saved text file
geometry_msgs/PoseStampedwithCovariance to match groundtruth.txt


GIVEN FORMAT:
    time(ns), field.header.seq, field.header.stamp, field.header.frame_id, x, y, z, qx, qy, qz, qw, field.pose.covariance0,field.pose.covariance1,field.pose.covariance2,field.pose.covariance3,field.pose.covariance4,field.pose.covariance5,field.pose.covariance6,field.pose.covariance7,field.pose.covariance8,field.pose.covariance9,field.pose.covariance10,field.pose.covariance11,field.pose.covariance12,field.pose.covariance13,field.pose.covariance14,field.pose.covariance15,field.pose.covariance16,field.pose.covariance17,field.pose.covariance18,field.pose.covariance19,field.pose.covariance20,field.pose.covariance21,field.pose.covariance22,field.pose.covariance23,field.pose.covariance24,field.pose.covariance25,field.pose.covariance26,field.pose.covariance27,field.pose.covariance28,field.pose.covariance29,field.pose.covariance30,field.pose.covariance31,field.pose.covariance32,field.pose.covariance33,field.pose.covariance34,field.pose.covariance35
    i.e. want columns: [0, 4, 5, 6, 7, 8, 9, 10]

DESIRED FORMAT:
    time(ns) x y z qx qy qz qw
"""

import sys
import numpy as np


def process_one_output_line(line, desired_idx):
    line = line.split(',')
    line = np.array(line)[desired_idx]
    line = line.astype(np.float)
    return line


def main(raw_file_name,GT_file_name):

    # Read files
    with open(raw_file_name) as f:
        raw_lines = f.readlines()[1:] # list of lists, one list per line

    gt = np.loadtxt(open(GT_file_name), delimiter=" ", skiprows=0)
    gt[:,1:4] -= gt[0,1:4] # get relative distances from starting point

    # Parse desired cols as floats
    desired_idx = [0, 4, 5, 6, 7, 8, 9, 10]
    processed_raw = np.array([process_one_output_line(line,desired_idx) for line in raw_lines])
    processed_raw[:,0] /= 1e9 # convert to nanoseconds

    # Match raw file lines to closes GT lines (by timestamp, nansoeconds)
    # build 2d difference matrix:
    # each row m is each timestamp in raw, each col n is each timestamp in gt
    raw_timestamps = processed_raw[:,0][:,np.newaxis] # (m,1)
    GT_timestamps = gt[:,0][:,np.newaxis].T # (1,n)
    diff = np.abs(raw_timestamps - GT_timestamps) # (m, n)
    matching_GT_idx = diff.argmin(axis = 1)
    matching_GT = gt[matching_GT_idx]
        
    # Compute pose distance between matching rows
    # position: L2
    # orientation: angle dist
    xyz_distance = (matching_GT - processed_raw)[:,1:4]
    norm = np.linalg.norm(xyz_distance, axis = 1)
    print(np.mean(norm))

    matching_GT_quats = matching_GT[:, 4:]
    raw_quats = processed_raw[:, 4:]
    cosine_angles = np.sum(matching_GT_quats * raw_quats, axis=1) / (np.linalg.norm(raw_quats, axis=1) * np.linalg.norm(matching_GT_quats, axis=1))
    angle_dists = 2*np.arccos(cosine_angles)
    print(matching_GT_quats[0])
    print(raw_quats[0])
    print(angle_dists)

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("USAGE:\nargs: raw_file.txt ground_truth.txt")
    else:
        raw_file_name = sys.argv[1]
        GT_file_name = sys.argv[2] # ground truth
        main(raw_file_name, GT_file_name)