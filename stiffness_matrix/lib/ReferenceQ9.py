# -*- coding: utf-8 -*-

# Import packages
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import scipy.linalg
import math
import os
import shutil
import zipfile
import glob

from tqdm import tqdm


class Q9_ElementAnalysis:
    def __init__(self, dir_save):
        print("{0:*^49}".format(""))
        print("* {0:^45} *".format("Q9 Elements Analysis Tool"))
        print("{0:*^49}".format(""))
        self.dir_save = dir_save
        pass

    # ******************************************************************
    #  Methods specific to Quadrangles elements
    # ******************************************************************

    def Q9_IsoparametricElement(self, fig_out: bool = True):
        self.E = 1  # Young’s modulus
        self.ν = 0.3  # Poisson’s ratio
        self.p = 1  # desity
        self.thick = 1  # thickness

        C = self.C_plane_strain()

        Q9_init = [
            (0, 0),
            (1, 0),
            (1.5, 1.0),
            (0.2, 1.5),
            (0.5, 0),
            (1.25, 0.5),
            (0.85, 1.25),
            (0.1, 0.75),
            (0.675, 0.625),
        ]
        """
        4 --- 7 --- 3
        |           |
        8     9     6
        |           |
        1 --- 5 --- 2
        """
        Q9 = [(x + 8, y + 8) for x, y in Q9_init]

        Q9_geo = np.asarray(Q9, dtype=float)  # (9,2)

        x = Q9_geo[:, 0]  # 모든 x
        y = Q9_geo[:, 1]  # 모든 y
        dof = 9
        node_idx = np.arange(len(x))  # node idx

        ELEM = np.column_stack([node_idx, x, y])  # (4, 3)

        K_global = np.zeros((2 * len(x), 2 * len(y)))

        for i in range(len(node_idx)):

            K_local_gauss, Mass_local_gauss = self.global_Mass_stiffness(
                ELEM=ELEM, idx=i, C=C
            )

        # print(np.shape(K_local_gauss))

        GK = K_local_gauss
        GM = Mass_local_gauss
        GF = np.ones((18, 18))

        # print(np.shape(GU))
        eigvals, eigvecs = scipy.linalg.eig(GK, GM)
        eigvecs = np.real(eigvecs)
        eigvals = np.real(eigvals)

        # 내림차순 정렬 인덱스
        idx = np.argsort(eigvals)[::-1]  # 큰 값 → 작은 값 순서

        # 고유값과 고유벡터 재정렬
        eigvals_sorted = eigvals[idx]
        eigvecs_sorted = eigvecs[:, idx]

        # # print(np.shape(eigvals_sorted))
        # print(eigvals_sorted)
        if fig_out:
            from .plot_Q9mode import plot_eigenmode

            plot_eigenmode(
                Q9_init=Q9_init,
                eigvecs=eigvecs_sorted,
                eigvals=eigvals_sorted,
                dir_save=self.dir_save,
            )

            print("{0:*^49}".format(""))
            print(" Q9 Isoparametric Element")
            print("{0:*^49}".format(""))

        return

    def C_plane_strain(self):
        E = self.E
        nu = self.ν
        if not (-1.0 < nu < 0.5):
            raise ValueError("Poisson's ratio nu must be in (-1, 0.5) for stability.")
        coef = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        C = coef * np.array(
            [
                [1 - nu, nu, 0],  # ε_x
                [nu, 1 - nu, 0],  # ε_y
                [0, 0, 0.5 * (1 - 2 * nu)],  # γ_xy
            ],
            dtype=float,
        )

        return C
        # -------------------------------

    # 사각형 p차 Lagrange 기저 (Gmsh 노드 순서)
    # -------------------------------
    def global_Mass_stiffness(self, ELEM, idx, C):
        p = self.p
        t = self.E
        c = self.ν

        K_local = np.zeros((9, 9))
        K_local_gauss = np.zeros((9, 9))

        x = ELEM[:, 1]

        y = ELEM[:, 2]
        # Gauss points/weights
        rr = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        ss = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        ww = [5 / 9, 8 / 9, 5 / 9]
        H = []

        for i in range(len(rr)):
            r = rr[i]
            s = ss[i]
            w = ww[i]

            h1 = 0.25 * (r - 1) * (s - 1) * r * s  # Node 1 (bottom-left)
            h2 = 0.25 * (r + 1) * (s - 1) * r * s  # Node 2 (bottom-right)
            h3 = 0.25 * (r + 1) * (s + 1) * r * s  # Node 3 (top-right)
            h4 = 0.25 * (r - 1) * (s + 1) * r * s  # Node 4 (top-left)
            h5 = 0.5 * (1 - r**2) * (s - 1) * s  # Node 5 (bottom edge)
            h6 = 0.5 * (r + 1) * r * (1 - s**2)  # Node 6 (right edge)
            h7 = 0.5 * (1 - r**2) * (s + 1) * s  # Node 7 (top edge)
            h8 = 0.5 * (r - 1) * r * (1 - s**2)  # Node 8 (left edge)
            h9 = (1 - r**2) * (1 - s**2)  # Node 9 (center)
            H = np.array([[h1, h2, h3, h4, h5, h6, h7, h8, h9]])

            # Strain-displacement matrix
            #  derivatives of interpolation functions wrt. r & s
            h1_r = 0.25 * s * (2 * r - 1) * (s - 1)  # Node 1
            h1_s = 0.25 * r * (r - 1) * (2 * s - 1)
            h2_r = 0.25 * s * (2 * r + 1) * (s - 1)  # Node 2
            h2_s = 0.25 * r * (r + 1) * (2 * s - 1)
            h3_r = 0.25 * s * (2 * r + 1) * (s + 1)
            h3_s = 0.25 * r * (r + 1) * (2 * s + 1)
            h4_r = 0.25 * s * (2 * r - 1) * (s + 1)  # Node 4
            h4_s = 0.25 * r * (r - 1) * (2 * s + 1)

            h5_r = 0.5 * (2 * r + 1) * (1 - s**2)  # Node 6 (right edge)
            h5_s = -s * (r + 1) * r
            h6_r = 0.5 * (1 - r**2) * (2 * s - 1)
            h6_s = 0.5 * (2 * r + 1) * (1 - s**2)  # Node 6 (right edge)
            h7_r = -s * (r + 1) * r
            h7_s = -r * (s + 1) * s  # Node 7 (top edge)
            h8_r = 0.5 * (2 * r - 1) * (1 - s**2)  # Node 8 (left edge)
            h8_s = -s * (r - 1) * r
            h9_r = -2 * r * (1 - s**2)  # Node 9
            h9_s = -2 * s * (1 - r**2)

            h_r = np.array(
                [(h1_r, h2_r, h3_r, h4_r, h5_r, h6_r, h7_r, h8_r, h9_r)]
            )  # 넘파이 배열

            h_s = np.array([(h1_s, h2_s, h3_s, h4_s, h5_s, h6_s, h7_s, h8_s, h9_s)])

            J = np.zeros((2, 2))

            # dx/dr, dy/dr, dx/ds, dy/ds
            J[0, 0] = np.dot(h_r, x)  # dx/dr
            J[0, 1] = np.dot(h_r, y)  # dy/dr
            J[1, 0] = np.dot(h_s, x)  # dx/ds
            J[1, 1] = np.dot(h_s, y)  # dy/ds

            # J[0, 0] = np.dot(h_r, x)  # dx/dr
            # J[0, 1] = np.dot(h_s, x)  # dy/dr
            # J[1, 0] = np.dot(h_r, y)  # dx/ds
            # J[1, 1] = np.dot(h_s, y)  # dy/ds

            # J = np.zeros((2, 8))
            # J[0, 0:4] = h_r * x
            # J[0, 4:8] = h_r * y
            # J[1, 0:4] = h_s * x
            # print(J)
            # J[1, 4:8] = h_s * y
            # # J = np.block([[h_r * x, h_s * x], [h_r * y, h_s * y]])

            # J = [ dx/dr , dy/dr ] 8

            J_inv = np.linalg.pinv(J)

            # Singular한 Jacobian에 대해 Moore-Penrose

            a = np.vstack([h_r, h_s])
            h_xy = J_inv @ a  # (2, nnode)

            h_x = h_xy[0, :]  # (nnode,)

            h_y = h_xy[1, :]  # (nnode,)

            # --- 표준 평면변형률(plane strain/stress) B 행렬(3 x 2n) -조립 ---

            # B = np.zeros((4, 8))

            B = np.zeros((3, 9 * 2))  # plane strain/stress 3x8

            # u1..u4, v1..v4 순서
            # B[0, 0:4] = h_x  # ε_x = ∂u/∂x
            # B[1, 4:8] = h_y  # ε_y = ∂v/∂y
            # B[2, 0:4] = h_x  # γ_xy = ∂u/∂y
            # B[2, 4:8] = h_y  # γ_xy = ∂v/∂x
            B[0, 0:9] = h_x  # ε_x = ∂u/∂x
            B[1, 9:18] = h_y  # ε_y = ∂v/∂y
            B[2, 0:9] = h_y  # γ_xy = ∂u/∂y
            B[2, 9:18] = h_x  # γ_xy = ∂v/∂x

            # B[0, 0:4] = h_x  # ε_x = ∂u/∂x
            # B[1, 4:8] = h_y  # ε_y = ∂v/∂y
            # B[3, 0:4] = h_y  # γ_xy = ∂u/∂y
            # B[3, 4:8] = h_x  # γ_xy = ∂v/∂x
            # B[3, 0:4] = h_y  # γ_xy = ∂u/∂y
            # B[3, 4:8] = h_x  # γ_xy = ∂v/∂x

            #  B (438) =
            # [ dN1/dx  dN2/dx  dN3/dx  dN4/dx    0       0       0       0   ]  → ε_x
            # [ 0       0       0       0        dN1/dy dN2/dy dN3/dy dN4/dy]  → ε_y
            # [ dN1/dy  dN2/dy  dN3/dy  dN4/dy   dN1/dx dN2/dx dN3/dx dN4/dx] → γ_xy
            # [ dN1/dy  dN2/dy  dN3/dy  dN4/dy   dN1/dx dN2/dx dN3/dx dN4/dx] → γ_xy

            """
                    u1 u2 u3 u4 v1 v2 v3 v4 
              u1[ K11 K12 K13 K14 K15 K16 K17 K18 ] 
              u2 [ K21 K22 K23 K24 K25 K26 K27 K28 ] 
              u3 [ K31 K32 K33 K34 K35 K36 K37 K38 ] 
              u4 [ K41 K42 K43 K44 K45 K46 K47 K48 ] 
              v1 [ K51 K52 K53 K54 K55 K56 K57 K58 ] 
              v2 [ K61 K62 K63 K64 K65 K66 K67 K68 ] 
              v3 [ K71 K72 K73 K74 K75 K76 K77 K78 ] 
              v4 [ K81 K82 K83 K84 K85 K86 K87 K88 ]

              대각 블록 (자기 자신)

            u1..u4 행 × u1..u4 열 → Node u 자유도 자기자신 stiffness

            v1..v4 행 × v1..v4 열 → Node v 자유도 자기자신 stiffness

            비대각 블록 (노드 간 상호작용)

            u1 행 × u2..u4 열 → Node 1 x 자유도가 Node 2,3,4에 미치는 stiffness

            u1 행 × v1..v4 열 → Node 1 x 자유도가 Node 1..4 y 자유도에 미치는 coupling

            v1 행 × u1..u4 열 → Node 1 y 자유도가 Node 1..4 x 자유도에 미치는 coupling

            v1 행 × v2..v4 열 → Node 1 y 자유도가 Node 2..4 y 자유도에 미치는 coupling
            """

            # print("C", np.shape(C))
            K_local = B.T @ C @ B * scipy.linalg.det(J)
            # Gauss integration
            K_local_gauss = 1 / 2 * K_local * w

            H2 = np.zeros((1, 9 * 2))
            H2[0, 0:9] = H  # u 방향
            H2[0, 9:18] = H  # v 방향
            Mass_local_gauss = p * t * H2.T @ H2

            # print(K_local_gauss)
        return K_local_gauss, Mass_local_gauss


if __name__ == "__main__":
    exit()
