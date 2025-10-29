import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import os
import shutil
import zipfile
import glob
import imageio

plt.style.use("./common/custom.mplstyle")


def plot_eigenmode(
    Q9_init,
    eigvecs,
    eigvals,
    dir_save: str = "./",
):

    # GIF 생성 설정
    n_frames = 50
    n_cycles = 2  # 4주기
    frames = []

    scale = 0.5  # 변위 스케일링
    coords = np.array(Q9_init, dtype=float)

    for mode_idx in range(18):
        mode = eigvecs[:, mode_idx]
        u = mode[0:9]
        v = mode[9:18]
        omega_i = np.sqrt(eigvals[mode_idx])  # 자연진동수

        filenames = []
        out_dir = dir_save or "eigenmodeQ9"
        os.makedirs(out_dir, exist_ok=True)

        # 선 연결 순서 (corners + edge midpoints)
        edge_order = [0, 4, 1, 5, 2, 6, 3, 7, 0]
        center_node = 8  # 9번 노드

        for i in range(n_frames):
            t = np.sin(2 * np.pi * n_cycles * i / n_frames)
            displaced = coords + scale * t * np.column_stack((u, v))

            fig, ax = plt.subplots(figsize=(10, 10))

            # undeformed
            ax.plot(
                coords[edge_order, 0],
                coords[edge_order, 1],
                "k--",
                label="undeformed",
            )
            # deformed
            ax.plot(
                displaced[edge_order, 0],
                displaced[edge_order, 1],
                "r-",
                lw=2,
                label="deformed",
            )

            # scatter 모든 노드
            ax.scatter(displaced[:, 0], displaced[:, 1], c="r")
            # 가운데 노드 강조
            ax.scatter(
                displaced[center_node, 0],
                displaced[center_node, 1],
                c="b",
                s=50,
                marker="o",
                label="center node (9)",
            )

            ax.set_xlim(-1, 2)
            ax.set_ylim(-1, 2)
            ax.set_aspect("equal")
            ax.set_title(f"Mode {mode_idx+1} (ω={omega_i:.3f})")
            ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5)
            ax.legend()

            # save frame
            filename = os.path.join(out_dir, f"mode{mode_idx+1}_frame{i}.png")
            plt.savefig(filename)
            plt.close(fig)
            filenames.append(filename)

        # GIF 생성
        gif_name = os.path.join(out_dir, f"mode_{mode_idx+1}.gif")
        with imageio.get_writer(gif_name, mode="I", duration=0.05) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                os.remove(filename)

    rigidvec = np.zeros((9 * 2, 3))
    # for mode_idx in range(3):

    #     rigidvec[:, 0] = [
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,  # u1~u9 노드 순서
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,  # v1~v9 (u 모드이므로 0)
    #     ]

    #     # v 방향 이동
    #     rigidvec[:, 1] = [
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,
    #         0,  # u1~u9
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,
    #         1,  # v1~v9
    #     ]  # v 방향 이동
    #     rigidvec[:, 2] = [
    #         0.5,
    #         0.5,
    #         -0.5,
    #         -0.5,
    #         0.5,
    #         0,
    #         -0.5,
    #         0,
    #         0,  # u1~u9
    #         -0.5,
    #         0.5,
    #         0.5,
    #         -0.5,
    #         0.0,
    #         0.5,
    #         0,
    #         -0.5,
    #         0,  # v1~v9
    #     ]
    # for mode_idx in range(3):
    #     mode = rigidvec[:, mode_idx]
    #     u = mode[0:9]
    #     v = mode[9:18]
    #     omega_i = np.sqrt(eigvals[mode_idx])  # 자연진동수

    #     filenames = []
    #     out_dir = dir_save or "eigenmodeQ9"
    #     os.makedirs(out_dir, exist_ok=True)

    #     # 선 연결 순서 (corners + edge midpoints)
    #     edge_order = [0, 4, 1, 5, 2, 6, 3, 7, 0]
    #     center_node = 8  # 9번 노드

    #     for i in range(n_frames):
    #         t = np.sin(2 * np.pi * n_cycles * i / n_frames)
    #         displaced = coords + scale * t * np.column_stack((u, v))

    #         fig, ax = plt.subplots(figsize=(10, 10))

    #         # undeformed
    #         ax.plot(
    #             coords[edge_order, 0],
    #             coords[edge_order, 1],
    #             "k--",
    #             label="undeformed",
    #         )
    #         # deformed
    #         ax.plot(
    #             displaced[edge_order, 0],
    #             displaced[edge_order, 1],
    #             "r-",
    #             lw=2,
    #             label="deformed",
    #         )

    #         # scatter 모든 노드
    #         ax.scatter(displaced[:, 0], displaced[:, 1], c="r")
    #         # 가운데 노드 강조
    #         ax.scatter(
    #             displaced[center_node, 0],
    #             displaced[center_node, 1],
    #             c="b",
    #             s=50,
    #             marker="o",
    #             label="center node (9)",
    #         )

    #         ax.set_xlim(-1, 2)
    #         ax.set_ylim(-1, 2)
    #         ax.set_aspect("equal")
    #         ax.set_title(f"Mode {mode_idx+1} (ω={omega_i:.3f})")
    #         ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5)
    #         ax.legend()

    #         # save frame
    #         filename = os.path.join(out_dir, f"mode{mode_idx+1}_frame{i}.png")
    #         plt.savefig(filename)
    #         plt.close(fig)
    #         filenames.append(filename)

    #     # GIF 생성
    #     gif_name = os.path.join(out_dir, f"rigidmod_{mode_idx+1}.gif")
    #     with imageio.get_writer(gif_name, mode="I", duration=0.05) as writer:
    #         for filename in filenames:
    #             image = imageio.imread(filename)
    #             writer.append_data(image)
    #             os.remove(filename)
