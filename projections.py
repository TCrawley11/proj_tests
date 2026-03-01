import numpy as np
import matplotlib.pyplot as plt

# intrinsic matrix structure:
# 0 0 -> focal length x 
# 0 2 -> principal point x
# 1 1 -> focal length y
# 1 2 -> principal point y

def pointcloud_to_pixel(K, pc_arr):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    x, y, z = pc_arr[:, 0], pc_arr[:, 1], pc_arr[:, 2]

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    return np.stack([u, v], axis=1)  # (N, 2)

def rgbd_to_pointcloud(K, rgbd_arr):
    # back-project pixels to 3d using depth via K^-1
    h, w = rgbd_arr.shape[0], rgbd_arr.shape[1]

    # homogeneous pixel coordinate grid: shape (3, H*W)
    vv, uu = np.indices((h, w))
    ones   = np.ones((h, w))
    pixels = np.stack([uu, vv, ones], axis=0).reshape(3, -1)

    # K^-1 @ pixels gives normalised ray directions, scale by depth
    K_inv = np.linalg.inv(K)
    xyz   = (K_inv @ pixels) * rgbd_arr.ravel()  # (3, H*W)

    valid = rgbd_arr.ravel() > 0
    return xyz.T[valid]  # (N, 3)

def get_dir_vec(K, row, col):
    # un-project pixel to a unit ray direction in camera space
    K_inv = np.linalg.inv(K)
    pixel = np.array([col, row, 1.0])
    d = K_inv @ pixel
    return d / np.linalg.norm(d)  # unit vector

def graph_dir_vecs(K, h, w, step=20):
    """
    Visualise the ray directions for every pixel on a subsampled grid.
    Arrows show the (X, Y) deviation of each ray from the optical axis.
    The principal point should have a zero-length arrow (ray points straight forward).

    K:    intrinsic matrix (3, 3)
    h, w: image height and width
    step: pixel stride for subsampling
    """
    K_inv = np.linalg.inv(K)

    rows = np.arange(0, h, step)
    cols = np.arange(0, w, step)
    uu, vv = np.meshgrid(cols, rows)          # (nrows, ncols)

    ones   = np.ones_like(uu, dtype=float)
    pixels = np.stack([uu, vv, ones], axis=0).reshape(3, -1)  # (3, N)

    dirs = K_inv @ pixels                     # (3, N) — unnormalised rays
    # normalise each column
    dirs /= np.linalg.norm(dirs, axis=0)

    dx = dirs[0].reshape(uu.shape)            # X component
    dy = dirs[1].reshape(vv.shape)            # Y component

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1) Quiver plot — arrows fan outward from the principal point
    axes[0].quiver(uu, vv, dx, dy, angles='xy', scale_units='xy', scale=0.05)
    axes[0].set_xlim(0, w)
    axes[0].set_ylim(h, 0)          # image convention: row 0 at top
    axes[0].set_title("Ray directions (X, Y components)")
    axes[0].set_xlabel("u (col)")
    axes[0].set_ylabel("v (row)")
    axes[0].set_aspect("equal")

    # 2) Colour map of ray angle from optical axis (field-of-view map)
    angle = np.degrees(np.arccos(dirs[2].reshape(uu.shape)))  # angle from Z axis
    im = axes[1].imshow(angle, extent=[0, w, h, 0], origin='upper', cmap='plasma')
    plt.colorbar(im, ax=axes[1], label="angle from optical axis (deg)")
    axes[1].set_title("Per-pixel ray angle (FoV map)")
    axes[1].set_xlabel("u (col)")
    axes[1].set_ylabel("v (row)")

    plt.tight_layout()
    plt.savefig("dir_vecs.png")
    plt.show()

def graph_comparison_pc_to_img(img_arr, pc_arr, gt_img_arr):
    H, W = gt_img_arr.shape[:2]

    u = np.round(img_arr[:, 0]).astype(int)
    v = np.round(img_arr[:, 1]).astype(int)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_v, v_v = u[valid], v[valid]

    rgb = pc_arr[:, 3:6].astype(np.uint8)
    proj_img = np.zeros((H, W, 3), dtype=np.uint8)
    proj_img[v_v, u_v] = rgb[valid]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    axes[0].imshow(gt_img_arr)
    axes[0].set_title("Ground Truth Image")
    axes[0].axis("off")

    axes[1].imshow(proj_img)
    axes[1].set_title("Projected Pointcloud Image")
    axes[1].axis("off")

    plt.savefig('comparison.png')
    plt.tight_layout()
    plt.show()

def graph_comparison_img_to_pc(pc_arr, img_arr, gt_pc_arr):
    """
    pc_arr:    back-projected pointcloud, shape (N, 3) — output of pixel_to_pointcloud
    img_arr:   input color image used for back-projection, shape (H, W, 3) uint8
    gt_pc_arr: ground truth pointcloud, shape (M, 6) — [x, y, z, r, g, b]
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1) Input image (ground truth for this direction)
    axes[0].imshow(img_arr)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # 2) Back-projected pointcloud — top-down X-Z, coloured by depth
    x_bp = pc_arr[:, 0]
    z_bp = pc_arr[:, 2]
    step_bp = max(1, len(pc_arr) // 50_000)
    sc = axes[1].scatter(
        x_bp[::step_bp], z_bp[::step_bp],
        c=z_bp[::step_bp], cmap="viridis",
        s=0.5, linewidths=0,
    )
    plt.colorbar(sc, ax=axes[1], label="Z / depth (m)")
    axes[1].set_title("Back-projected Pointcloud")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z / depth (m)")
    axes[1].set_aspect("equal")

    # 3) Ground truth pointcloud — top-down X-Z, coloured by point RGB
    x_gt = gt_pc_arr[:, 0]
    z_gt = gt_pc_arr[:, 2]
    colors_gt = gt_pc_arr[:, 3:6] / 255.0
    step_gt = max(1, len(gt_pc_arr) // 50_000)
    axes[2].scatter(
        x_gt[::step_gt], z_gt[::step_gt],
        c=colors_gt[::step_gt],
        s=0.5, linewidths=0,
    )
    axes[2].set_title("Ground Truth Pointcloud")
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Z / depth (m)")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("comparison_img_to_pc.png")
    plt.show()

def main():
    color_file = "example_color.npy"
    depth_file = "example_depth.npy"
    pc_file    = "example_pointcloud.npy"
    intr_file  = "intrinsic_matrix.npy"

    intr_arr  = np.load(intr_file)
    color_arr = np.load(color_file)
    depth_arr = np.load(depth_file)
    pc_arr    = np.load(pc_file)

    #proj_pixels = pointcloud_to_pixel(intr_arr, pc_arr)
    #graph_comparison_pc_to_img(proj_pixels, pc_arr, color_arr)

    #proj_pc = rgbd_to_pointcloud(intr_arr, depth_arr)
    #graph_comparison_img_to_pc(proj_pc, color_arr, pc_arr)

    h, w = depth_arr.shape
    graph_dir_vecs(intr_arr, h, w)

if __name__ == "__main__":
    main()

