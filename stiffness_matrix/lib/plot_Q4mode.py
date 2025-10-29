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
    Q4_init,
    node_idx,
    eigvals,
    eigvecs,
    dir_save: str = "./",
):

    # GIF 생성 설정
    n_frames = 50
    coords = np.array(Q4_init, dtype=float)  # (4,2)
    t_vals = np.linspace(0, 4 * np.pi, n_frames)  # 1주기

    scale = 0.5  # 변위 스케일링
    coords = np.array(Q4_init, dtype=float)

    for mode_idx in range(8):

        mode = eigvecs[:, mode_idx]  # mode_idx 번째 모드
        u = mode[0:4]
        v = mode[4:8]
        # print(np.shape(u))
        # print(u)
        omega_i = np.sqrt(eigvals[mode_idx])  # 자연진동수
        # omega_i = 1

        filenames = []

        # print((eigvecs))

        out_dir = dir_save
        if not out_dir:
            out_dir = f"eigenmodeQ4"
            dir_save = out_dir
        os.makedirs(out_dir, exist_ok=True)  # ← 폴더 보장
        filenames = []

        for i, t in enumerate(t_vals):
            # 변위 적용: sin(ωt)
            displaced = coords + scale * np.column_stack((u, v)) * np.sin(1 * t)

            # 플로팅
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(
                coords[[0, 1, 2, 3, 0], 0],
                coords[[0, 1, 2, 3, 0], 1],
                "k--",
                label="undeformed",
            )
            ax.plot(
                displaced[[0, 1, 2, 3, 0], 0],
                displaced[[0, 1, 2, 3, 0], 1],
                "r-",
                lw=2,
                label="deformed",
            )
            ax.scatter(displaced[:, 0], displaced[:, 1], c="r")
            ax.set_xlim(-1, 2)
            ax.set_ylim(-1, 2)
            ax.set_aspect("equal")
            ax.set_title(f"Mode {mode_idx+1} (ω={omega_i:.3f})")
            ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5)

            filename = os.path.join(out_dir, f"mode{mode_idx+1}_frame_{i:03d}.png")
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

    # GIF 생성 설정

    # rigidvec = np.zeros((8, 3))
    # for mode_idx in range(3):
    #     rigidvec[:, 0] = [1, 1, 1, 1, 0, 0, 0, 0]  # u,v 방향 이동
    #     rigidvec[:, 1] = [0, 0, 0, 0, 1, 1, 1, 1]  # v 방향 이동
    #     rigidvec[:, 2] = [0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5]  # 회전
    #     # print(np.shape(rigidvec))
    #     # print((rigidvec))
    #     mode = rigidvec[:, mode_idx]
    #     u = mode[0:4]
    #     v = mode[4:8]
    #     omega_i = 1  # 자연진동수

    #     # print((eigvecs))

    #     # 저장 디렉터리 설정(예: svg_Line_elements_order_4_Lagrange)
    #     out_dir = dir_save
    #     if not out_dir:
    #         out_dir = f"eigenmodeQ4"
    #         dir_save = out_dir
    #     os.makedirs(out_dir, exist_ok=True)  # ← 폴더 보장
    #     filenames = []

    #     for i, t in enumerate(t_vals):
    #         # 변위 적용: sin(ωt)
    #         displaced = coords + scale * np.column_stack((u, v)) * np.sin(omega_i * t)

    #         # 플로팅
    #         fig, ax = plt.subplots(figsize=(8, 8))
    #         ax.plot(
    #             coords[[0, 1, 2, 3, 0], 0],
    #             coords[[0, 1, 2, 3, 0], 1],
    #             "k--",
    #             label="undeformed",
    #         )
    #         ax.plot(
    #             displaced[[0, 1, 2, 3, 0], 0],
    #             displaced[[0, 1, 2, 3, 0], 1],
    #             "r-",
    #             lw=2,
    #             label="deformed",
    #         )
    #         ax.scatter(displaced[:, 0], displaced[:, 1], c="r")
    #         ax.set_xlim(-1, 2)
    #         ax.set_ylim(-1, 2)
    #         ax.set_aspect("equal")
    #         ax.set_title(f"Mode {mode_idx+1} (ω={'nan'})")
    #         ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5)

    #         filename = os.path.join(out_dir, f"mode{mode_idx+1}_frame_{i:03d}.png")
    #         plt.savefig(filename)
    #         plt.close(fig)
    #         filenames.append(filename)

    #     # GIF 생성
    #     gif_name = os.path.join(out_dir, f"rigidmode_{mode_idx+1}.gif")
    #     with imageio.get_writer(gif_name, mode="I", duration=0.05) as writer:
    #         for filename in filenames:
    #             image = imageio.imread(filename)
    #             writer.append_data(image)
    #             os.remove(filename)
