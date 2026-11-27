# -*- coding: utf-8 -*-
"""URDF 正运动学解算、位姿矩阵变换及本地网格打包工具模块。"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import FILE_IO_CHUNK_BYTES, MESH_COPY_WORKERS
from .lerobot_spec import FrameCameraExtrinsics, MeshFileDigest, TransformPose, UrdfJointInfo


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """将 URDF 固定 rpy 欧拉角（弧度）转换为 3x3 旋转矩阵。

    参数:
        roll: 绕 X 轴滚转角（弧度）。
        pitch: 绕 Y 轴俯仰角（弧度）。
        yaw: 绕 Z 轴偏航角（弧度）。
    返回值:
        3x3 float64 旋转矩阵。
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def axis_angle_to_matrix(axis_x: float, axis_y: float, axis_z: float, angle: float) -> np.ndarray:
    """将轴角旋转转换为 3x3 旋转矩阵。

    参数:
        axis_x: 旋转轴 X 分量。
        axis_y: 旋转轴 Y 分量。
        axis_z: 旋转轴 Z 分量。
        angle: 旋转角（弧度）。
    返回值:
        3x3 float64 旋转矩阵。
    """
    norm = math.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis_x / norm, axis_y / norm, axis_z / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """将归一化四元数 [qx, qy, qz, qw] 转换为 3x3 旋转矩阵。

    参数:
        qx: 四元数 X 分量。
        qy: 四元数 Y 分量。
        qz: 四元数 Z 分量。
        qw: 四元数 W 分量。
    返回值:
        3x3 float64 旋转矩阵。
    """
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(mat: np.ndarray) -> Tuple[float, float, float, float]:
    """将 3x3 旋转矩阵转换为归一化四元数 (qx, qy, qz, qw)。

    参数:
        mat: 3x3（或 4x4 左上角）旋转矩阵。
    返回值:
        归一化后的四元数元组 (qx, qy, qz, qw)。
    """
    m = mat[:3, :3]
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def matrix_to_transform_pose(mat: np.ndarray) -> TransformPose:
    """将 4x4 齐次变换矩阵转换为 TransformPose 对象。

    参数:
        mat: 4x4 齐次变换矩阵。
    返回值:
        TransformPose 位姿对象。
    """
    qx, qy, qz, qw = matrix_to_quat_xyzw(mat[:3, :3])
    return TransformPose(
        tx=float(mat[0, 3]),
        ty=float(mat[1, 3]),
        tz=float(mat[2, 3]),
        qw=float(qw),
        qx=float(qx),
        qy=float(qy),
        qz=float(qz),
    )


def parse_urdf_joints(urdf_xml: str) -> List[UrdfJointInfo]:
    """解析 URDF XML 中的全部关节定义与限位参数。

    参数:
        urdf_xml: URDF XML 字符串内容。
    返回值:
        UrdfJointInfo 对象列表。
    """
    root = ET.fromstring(urdf_xml)
    joints: List[UrdfJointInfo] = []
    for elem in root.findall("joint"):
        name = elem.attrib["name"]
        jtype = elem.attrib["type"]
        parent_link = elem.find("parent").attrib["link"]
        child_link = elem.find("child").attrib["link"]
        orig_el = elem.find("origin")
        if orig_el is not None:
            xyz_vals = [float(x) for x in orig_el.attrib.get("xyz", "0 0 0").split()]
            rpy_vals = [float(x) for x in orig_el.attrib.get("rpy", "0 0 0").split()]
        else:
            xyz_vals = [0.0, 0.0, 0.0]
            rpy_vals = [0.0, 0.0, 0.0]
        axis_el = elem.find("axis")
        if axis_el is not None:
            axis_vals = [float(x) for x in axis_el.attrib.get("xyz", "0 0 1").split()]
        else:
            axis_vals = [0.0, 0.0, 1.0]
        lim_el = elem.find("limit")
        if lim_el is not None:
            lower = float(lim_el.attrib.get("lower", "0.0"))
            upper = float(lim_el.attrib.get("upper", "0.0"))
            effort = float(lim_el.attrib.get("effort", "0.0"))
            velocity = float(lim_el.attrib.get("velocity", "0.0"))
        else:
            lower, upper, effort, velocity = 0.0, 0.0, 0.0, 0.0
        joints.append(
            UrdfJointInfo(
                name=name,
                joint_type=jtype,
                parent_link=parent_link,
                child_link=child_link,
                origin_xyz=(xyz_vals[0], xyz_vals[1], xyz_vals[2]),
                origin_rpy=(rpy_vals[0], rpy_vals[1], rpy_vals[2]),
                axis_xyz=(axis_vals[0], axis_vals[1], axis_vals[2]),
                lower=lower,
                upper=upper,
                effort=effort,
                velocity=velocity,
            )
        )
    return joints


class UrdfKinematicsSolver:
    """基于 URDF 关节链的轻量级正运动学（FK）解算器。"""

    def __init__(self, joints: List[UrdfJointInfo]) -> None:
        """初始化运动学解算器。

        参数:
            joints: 从 URDF 解析出的关节列表。
        """
        self._joints_by_child = {j.child_link: j for j in joints}
        self._joints_by_name = {j.name: j for j in joints}

    def get_joint_limit(self, joint_name: str) -> Optional[UrdfJointInfo]:
        """按关节名称查询限位信息。

        参数:
            joint_name: URDF 关节名。
        返回值:
            匹配的 UrdfJointInfo，不存在时返回 None。
        """
        return self._joints_by_name.get(joint_name)

    def _compute_chain_from_base(
        self,
        target_link: str,
        leg_q: np.ndarray,
        head_q: np.ndarray,
        left_arm_q: np.ndarray,
        right_arm_q: np.ndarray,
    ) -> np.ndarray:
        """计算从 base_link 到 target_link 的 4x4 齐次变换矩阵。"""
        q_lookup = {}
        for i in range(5):
            q_lookup[f"leg_joint{i + 1}"] = float(leg_q[i])
        for i in range(2):
            q_lookup[f"head_joint{i + 1}"] = float(head_q[i])
        for i in range(7):
            q_lookup[f"left_arm_joint{i + 1}"] = float(left_arm_q[i])
            q_lookup[f"right_arm_joint{i + 1}"] = float(right_arm_q[i])

        chain: List[UrdfJointInfo] = []
        curr = target_link
        while curr != "base_link":
            j = self._joints_by_child.get(curr)
            if j is None:
                raise KeyError(f"URDF 运动学链断裂，找不到父连杆: {curr}")
            chain.append(j)
            curr = j.parent_link

        mat = np.eye(4, dtype=np.float64)
        for j in reversed(chain):
            step = np.eye(4, dtype=np.float64)
            step[:3, :3] = rpy_to_matrix(*j.origin_rpy)
            step[:3, 3] = list(j.origin_xyz)
            if j.joint_type in ("revolute", "continuous"):
                angle = q_lookup.get(j.name, 0.0)
                rot = np.eye(4, dtype=np.float64)
                rot[:3, :3] = axis_angle_to_matrix(*j.axis_xyz, angle)
                step = step @ rot
            mat = mat @ step
        return mat

    def compute_end_effector_poses(
        self,
        leg_q: np.ndarray,
        head_q: np.ndarray,
        left_arm_q: np.ndarray,
        right_arm_q: np.ndarray,
    ) -> Tuple[TransformPose, TransformPose]:
        """计算左右臂末端法兰在 base_link 基座系下的位姿。

        参数:
            leg_q: 腿部 5 关节角度数组（弧度）。
            head_q: 头部 2 关节角度数组（弧度）。
            left_arm_q: 左臂 7 关节角度数组（弧度）。
            right_arm_q: 右臂 7 关节角度数组（弧度）。
        返回值:
            (左臂末端位姿, 右臂末端位姿)。
        """
        t_left = self._compute_chain_from_base(
            "left_arm_end_effector_mount_link", leg_q, head_q, left_arm_q, right_arm_q
        )
        t_right = self._compute_chain_from_base(
            "right_arm_end_effector_mount_link", leg_q, head_q, left_arm_q, right_arm_q
        )
        return matrix_to_transform_pose(t_left), matrix_to_transform_pose(t_right)

    def compute_camera_extrinsics(
        self,
        frame_index: int,
        leg_q: np.ndarray,
        head_q: np.ndarray,
        left_arm_q: np.ndarray,
        right_arm_q: np.ndarray,
        head_mount_T_head_left: TransformPose,
        stereo_left_T_right: TransformPose,
        left_mount_T_hand_left: TransformPose,
        right_mount_T_hand_right: TransformPose,
    ) -> FrameCameraExtrinsics:
        """计算单帧四路相机光学坐标系相对 base_link 的齐次变换。

        参数:
            frame_index: 当前帧索引。
            leg_q: 腿部关节角度（弧度）。
            head_q: 头部关节角度（弧度）。
            left_arm_q: 左臂关节角度（弧度）。
            right_arm_q: 右臂关节角度（弧度）。
            head_mount_T_head_left: 头部挂架到左目相机静态变换。
            stereo_left_T_right: 双目右目到左目基线变换。
            left_mount_T_hand_left: 左腕法兰到左腕相机静态变换。
            right_mount_T_hand_right: 右腕法兰到右腕相机静态变换。
        返回值:
            FrameCameraExtrinsics 对象。
        """
        t_head_mount = self._compute_chain_from_base(
            "head_end_effector_mount_link", leg_q, head_q, left_arm_q, right_arm_q
        )
        t_left_mount = self._compute_chain_from_base(
            "left_arm_end_effector_mount_link", leg_q, head_q, left_arm_q, right_arm_q
        )
        t_right_mount = self._compute_chain_from_base(
            "right_arm_end_effector_mount_link", leg_q, head_q, left_arm_q, right_arm_q
        )
        t_base_hl = t_head_mount @ head_mount_T_head_left.to_matrix()
        t_base_hr = t_base_hl @ stereo_left_T_right.to_matrix()
        t_base_hand_l = t_left_mount @ left_mount_T_hand_left.to_matrix()
        t_base_hand_r = t_right_mount @ right_mount_T_hand_right.to_matrix()
        return FrameCameraExtrinsics(
            frame_index=frame_index,
            base_T_head_stereo_left=matrix_to_transform_pose(t_base_hl),
            base_T_head_stereo_right=matrix_to_transform_pose(t_base_hr),
            base_T_hand_left=matrix_to_transform_pose(t_base_hand_l),
            base_T_hand_right=matrix_to_transform_pose(t_base_hand_r),
        )


def ensure_local_urdf_with_meshes(
    urdf_xml: str,
    source_desc_root: Path,
    dest_urdf_dir: Path,
    robot_type: str,
) -> Tuple[Path, List[MeshFileDigest]]:
    """保存本体 URDF 及全部引用网格到 meta/agibot/urdf/，同步计算 SHA256 并改写相对路径。

    参数:
        urdf_xml: 机器人本体 URDF XML 字符串。
        source_desc_root: 原始 galbot_one_golf_description 包根目录。
        dest_urdf_dir: 目标 meta/agibot/urdf 目录路径。
        robot_type: 本体型号名称，用于输出 <robot_type>.urdf。
    返回值:
        (生成的 URDF 路径, 网格摘要 MeshFileDigest 列表)。
    """
    dest_urdf_dir.mkdir(parents=True, exist_ok=True)
    root = ET.fromstring(urdf_xml)
    unique_rel_paths: List[str] = []
    seen = set()

    for mesh_el in root.findall(".//mesh"):
        raw_fn = mesh_el.attrib.get("filename", "")
        prefix = "package://galbot_one_golf_description/"
        rel_path = raw_fn[len(prefix) :] if raw_fn.startswith(prefix) else raw_fn
        normalized_rel = rel_path.replace("\\", "/")
        mesh_el.attrib["filename"] = normalized_rel
        if normalized_rel not in seen:
            seen.add(normalized_rel)
            unique_rel_paths.append(normalized_rel)
            # 扫描 .obj 引用的伴随 .mtl 材质文件并一并纳入交付与哈希清单
            if normalized_rel.lower().endswith(".obj"):
                src_obj = source_desc_root / normalized_rel
                if src_obj.exists():
                    obj_parent_rel = Path(normalized_rel).parent
                    with open(src_obj, "r", encoding="utf-8", errors="ignore") as f_obj:
                        for idx_line, line in enumerate(f_obj):
                            if idx_line >= 64:
                                break
                            stripped = line.strip()
                            if stripped.startswith("mtllib "):
                                mtl_name = stripped.split(None, 1)[1].strip()
                                mtl_rel = (obj_parent_rel / mtl_name).as_posix()
                                if (source_desc_root / mtl_rel).exists() and mtl_rel not in seen:
                                    seen.add(mtl_rel)
                                    unique_rel_paths.append(mtl_rel)

    def copy_and_hash_one(rel: str) -> MeshFileDigest:
        """单遍读取源网格文件并同步完成目标写入与内存 SHA256 计算。"""
        src_mesh = source_desc_root / rel
        dst_mesh = dest_urdf_dir / rel
        if not src_mesh.exists():
            raise FileNotFoundError(f"找不到 URDF 引用的网格文件: {src_mesh}")
        dst_mesh.parent.mkdir(parents=True, exist_ok=True)
        need_write = not dst_mesh.exists() or dst_mesh.stat().st_size != src_mesh.stat().st_size
        hasher = hashlib.sha256()
        if need_write:
            with open(src_mesh, "rb") as f_in, open(dst_mesh, "wb") as f_out:
                for chunk in iter(lambda: f_in.read(FILE_IO_CHUNK_BYTES), b""):
                    hasher.update(chunk)
                    f_out.write(chunk)
        else:
            with open(dst_mesh, "rb") as f_in:
                for chunk in iter(lambda: f_in.read(FILE_IO_CHUNK_BYTES), b""):
                    hasher.update(chunk)
        return MeshFileDigest(rel_path=rel, sha256=hasher.hexdigest())

    with ThreadPoolExecutor(max_workers=MESH_COPY_WORKERS) as pool:
        digests = list(pool.map(copy_and_hash_one, sorted(unique_rel_paths)))

    out_urdf_path = dest_urdf_dir / f"{robot_type}.urdf"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_urdf_path, encoding="utf-8", xml_declaration=True)
    return out_urdf_path, digests
