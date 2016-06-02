import numpy as np
import scipy.stats as st
import cv2

def gaussian_template(kernlen, nsig):
    interval = (2 * nsig + 1.0)/ (kernlen)
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = 2 * (kernel_raw / kernel_raw.sum())
    return kernel

def new_heatmap(w, h):
	return np.zeros((w, h))

def add_at_offset(b1, b2, pos_h, pos_v):
	# if pos_h + b2.shape[0] >= pos_b1.shape[0] or pos_v + b2.shape[1] >= pos_b1.shape[1]: return
	# if pos_h < 0 or pos_v < 0: return

	v_range1 = slice(max(0, pos_v), max(min(pos_v + b2.shape[0], b1.shape[0]), 0))
	h_range1 = slice(max(0, pos_h), max(min(pos_h + b2.shape[1], b1.shape[1]), 0))

	v_range2 = slice(max(0, -pos_v), min(-pos_v + b1.shape[0], b2.shape[0]))
	h_range2 = slice(max(0, -pos_h), min(-pos_h + b1.shape[1], b2.shape[1]))

	b1[v_range1, h_range1] += b2[v_range2, h_range2]

def update_heatmap(heatmap, additive_template, pose, decay):
	# add template the hands and feet
	left_hand_x = pose[30]
	left_hand_y = pose[31]
	add_at_offset(heatmap, additive_template, left_hand_x, left_hand_y)

	right_hand_x = pose[20]
	right_hand_y = pose[21]
	add_at_offset(heatmap, additive_template, right_hand_x, right_hand_y)

	left_foot_x = pose[10]
	left_foot_y = pose[11]
	add_at_offset(heatmap, additive_template, left_foot_x, left_foot_y)

	right_foot_x = pose[0]
	right_foot_y = pose[1]
	add_at_offset(heatmap, additive_template, right_foot_x, right_foot_y)

	# decay
	heatmap *= decay
