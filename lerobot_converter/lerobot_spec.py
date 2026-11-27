# -*- coding: utf-8 -*-
"""LeRobot v3.0 与 agibot 扩展元数据的强类型数据结构、MCAP 解析及 URDF 正运动学工具模块。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    GALBOT_GRIPPER_MAX_WIDTH_M,
    FILE_IO_CHUNK_BYTES,
    STATE_SOURCE as DEFAULT_STATE_SOURCE,
    TARGET_VIDEO_FPS,
    TARGET_VIDEO_HEIGHT,
    TARGET_VIDEO_WIDTH,
    VIDEO_CODEC as DEFAULT_VIDEO_CODEC,
    VIDEO_SOURCE as DEFAULT_VIDEO_SOURCE,
    X265_PRESET as DEFAULT_X265_PRESET,
)

# 规范化后的技能词表（对应规范 10.2 节）
STANDARD_SKILL_VOCAB: Tuple[str, ...] = (
    "Pick",
    "Pick up",
    "Place",
    "Put in",
    "Pour",
    "Open",
    "Close",
    "Insert",
    "Pull",
    "Clip",
    "PressButton",
    "PullApart",
    "Pull open",
    "PushTogether",
    "Push closed",
    "Straighten",
    "Lower",
    "OpenBox",
    "CloseBox",
    "pick-and-place",
)


@dataclass
class FieldDescription:
    """状态或动作向量的子字段语义描述。"""

    # 字段分层名称，如 state/left_arm/joint_pos
    key: str
    # 字段物理含义、坐标系与单位中文说明
    description: str
    # 占用向量维度数量
    dimensions: int
    # 在扁平向量中的连续索引下标列表
    indices: List[int]


@dataclass
class CameraSpec:
    """单路相机的源通道到标准语义键映射规格。"""

    # 标准语义特征键，如 observation.images.head_stereo_left
    feature_key: str
    # 简短相机键名，用于 camera_params.json 键名拼接
    short_name: str
    # 原始文件夹中的 mp4 文件名
    raw_mp4_name: str
    # MCAP 中的图像话题名
    image_topic: str
    # MCAP 中的相机内参话题名
    info_topic: str
    # TF 树中的光学坐标系名称
    optical_frame: str
    # 相机物理安装父连杆名称
    mount_parent_frame: str
    # 相机描述
    camera_desc: str


@dataclass
class CameraIntrinsics:
    """单路相机内参及畸变参数（已同步缩放至目标交付分辨率）。"""

    # 相机短名
    camera_name: str
    # 原始宽度（像素）
    src_width: int
    # 原始高度（像素）
    src_height: int
    # 目标宽度（像素）
    width: int
    # 目标高度（像素）
    height: int
    # 水平焦距（像素）
    fx: float
    # 垂直焦距（像素）
    fy: float
    # 主点横坐标（像素）
    cx: float
    # 主点纵坐标（像素）
    cy: float
    # 畸变模型名称（plumb_bob 或 kannala_brandt）
    distortion_model: str
    # 相机型号标识
    camera_type: str
    # 径向与切向畸变系数列表
    distortion_coeffs: List[float]


@dataclass
class TransformPose:
    """三维刚体位姿变换（平移单位米，四元数按 w/x/y/z 显式命名）。"""

    # 平移 X（米）
    tx: float
    # 平移 Y（米）
    ty: float
    # 平移 Z（米）
    tz: float
    # 四元数 W 分量
    qw: float
    # 四元数 X 分量
    qx: float
    # 四元数 Y 分量
    qy: float
    # 四元数 Z 分量
    qz: float

    def to_matrix(self) -> np.ndarray:
        """将位姿转换为 4x4 齐次变换矩阵。"""
        mat = np.eye(4, dtype=np.float64)
        norm = math.sqrt(self.qx * self.qx + self.qy * self.qy + self.qz * self.qz + self.qw * self.qw)
        if norm >= 1e-12:
            x, y, z, w = self.qx / norm, self.qy / norm, self.qz / norm, self.qw / norm
            mat[:3, :3] = np.array(
                [
                    [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                    [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                    [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
                ],
                dtype=np.float64,
            )
        mat[:3, 3] = [self.tx, self.ty, self.tz]
        return mat


@dataclass
class FrameCameraExtrinsics:
    """单帧各路相机相对于基座系的外参集合。"""

    # 帧序号
    frame_index: int
    # 头部左目相对基座变换 T_base_head_stereo_left
    base_T_head_stereo_left: TransformPose
    # 头部右目相对基座变换 T_base_head_stereo_right
    base_T_head_stereo_right: TransformPose
    # 左腕相机相对基座变换 T_base_hand_left
    base_T_hand_left: TransformPose
    # 右腕相机相对基座变换 T_base_hand_right
    base_T_hand_right: TransformPose


@dataclass
class UrdfJointInfo:
    """URDF 关节定义与限位信息。"""

    # 关节名称
    name: str
    # 关节类型（fixed / revolute / continuous）
    joint_type: str
    # 父连杆名称
    parent_link: str
    # 子连杆名称
    child_link: str
    # 关节安装位置平移 [x, y, z]（米）
    origin_xyz: Tuple[float, float, float]
    # 关节安装姿态欧拉角 [r, p, y]（弧度）
    origin_rpy: Tuple[float, float, float]
    # 旋转轴方向向量
    axis_xyz: Tuple[float, float, float]
    # 关节下限（弧度）
    lower: float
    # 关节上限（弧度）
    upper: float
    # 最大力矩（N·m）
    effort: float
    # 最大速度（rad/s）
    velocity: float


@dataclass
class SkillSegmentRecord:
    """L2 技能分段标注记录。"""

    # 所属 episode 编号
    episode_index: int
    # 规范化后的标准技能词（属于标准词表）
    skill: str
    # 中文动作描述
    label: str
    # 英文动作描述
    english_label: str
    # 起始帧索引（闭区间）
    start_frame_index: int
    # 结束帧索引（开区间）
    end_frame_index: int
    # 是否为失败/错误动作片段
    is_mistake: bool


@dataclass
class EpisodeSourceInfo:
    """单个原始采集 episode 的路径与元数据汇总。"""

    # 数据集内的连续 episode 编号
    episode_index: int
    # 原始样本文件夹路径
    source_dir: Path
    # 原始 MCAP 文件路径
    mcap_path: Path
    # 原始 episode ID
    episode_id: str
    # 32 位十六进制 episode 唯一标识
    episode_id_u32: str
    # 原始任务 ID
    task_id: int
    # 中文任务名
    task_name: str
    # 英文任务名
    english_task_name: str
    # 任务高层语言指令
    high_level_instruction: str
    # 机器人序列号
    sn_code: str
    # 采集机器人型号声明
    robot_type_raw: str
    # 硬件版本号
    hardware_version: str
    # 软件主线版本号
    sw_version: str
    # 技能分段标注列表
    segments: List[SkillSegmentRecord]


@dataclass
class McapJointSeries:
    """从 MCAP singorix/wbcs/sensor 提取的全通道关节时间序列。"""

    # 纳秒级时间戳序列 (M,)
    timestamps_ns: np.ndarray
    # 左臂关节位置 (M, 7) 与速度 (M, 7)
    left_arm_pos: np.ndarray
    left_arm_vel: np.ndarray
    # 右臂关节位置 (M, 7) 与速度 (M, 7)
    right_arm_pos: np.ndarray
    right_arm_vel: np.ndarray
    # 左右夹爪百分比行程 (M,)
    left_gripper_pct: np.ndarray
    right_gripper_pct: np.ndarray
    # 头部关节位置 (M, 2) 与速度 (M, 2)
    head_pos: np.ndarray
    head_vel: np.ndarray
    # 腿部关节位置 (M, 5) 与速度 (M, 5)
    leg_pos: np.ndarray
    leg_vel: np.ndarray


@dataclass
class McapOdomSeries:
    """从 MCAP /odom/base_link 提取的底盘里程计时间序列。"""

    # 纳秒级时间戳序列 (K,)
    timestamps_ns: np.ndarray
    # 平面位置 [x, y] (K, 2)
    pos_xy: np.ndarray
    # 归一化四元数 [qx, qy, qz, qw] (K, 4)
    ori_xyzw: np.ndarray


@dataclass
class EpisodeMcapParsed:
    """从单个 SYNC.mcap 中解析出的标定、URDF、时间戳及可选状态序列。"""

    # 机器人本体 URDF XML 文本
    urdf_xml: str
    # 机器序列号
    robot_serial_number: str
    # 硬件版本号
    hardware_version: str
    # 软件版本
    sw_version: str
    # 基准相机每帧 Unix 纳秒时间戳序列
    head_left_timestamps_ns: np.ndarray
    # 四路相机原始内参列表
    camera_intrinsics: List[CameraIntrinsics]
    # 双目基线变换（右目到左目：extrinsic_head_stereo_left_T_right）
    stereo_left_T_right: TransformPose
    # 头部挂架到左目光学系静态变换
    head_mount_T_head_left: TransformPose
    # 左腕末端法兰到左腕相机静态变换
    left_mount_T_hand_left: TransformPose
    # 右腕末端法兰到右腕相机静态变换
    right_mount_T_hand_right: TransformPose
    # MCAP 内提取的关节状态序列（未提取或无数据时为 None）
    joint_series: Optional[McapJointSeries] = None
    # MCAP 内提取的底盘里程计序列（未提取或无数据时为 None）
    odom_series: Optional[McapOdomSeries] = None


@dataclass
class MeshFileDigest:
    """URDF 网格文件的相对路径与 SHA256 摘要。"""

    # 相对 meta/agibot/urdf 的文件路径
    rel_path: str
    # 文件 SHA256 十六进制摘要
    sha256: str


@dataclass
class VideoCheckResult:
    """单路视频并发检查结果。"""

    # 所属 episode 编号
    episode_index: int
    # 相机简短名称
    camera_name: str
    # 解码与 PTS 对齐是否通过
    dec_ok: bool
    # 编码规格与 GOP=8 是否通过
    enc_ok: bool
    # 详细描述信息
    detail: str


@dataclass
class ValidationCheckItem:
    """单条验收检查结果。"""

    # 检查项优先级（P00 / P0 / P1）
    priority: str
    # 检查项名称
    name: str
    # 是否通过
    passed: bool
    # 详细说明
    detail: str


@dataclass
class ValidationReport:
    """完整数据集自检与验收报告。"""

    # 被检查数据集根目录
    dataset_root: str
    # 声明的数据分级档位
    data_tier: str
    # 全部检查项结果列表
    items: List[ValidationCheckItem] = field(default_factory=list)
    # 阻断级（P00 + P0）未通过计数
    blocking_issue_count: int = 0
    # P00 未通过计数
    p00_fail_count: int = 0
    # P0 未通过计数
    p0_fail_count: int = 0
    # P1 警告计数
    p1_warn_count: int = 0


# 四路标准相机映射配置
STANDARD_CAMERAS: Tuple[CameraSpec, ...] = (
    CameraSpec(
        feature_key="observation.images.head_stereo_left",
        short_name="head_stereo_left",
        raw_mp4_name="camera_head_left_color.mp4",
        image_topic="/front_head_camera/left_color/image_raw",
        info_topic="/front_head_camera/left_color/camera_info",
        optical_frame="head_left_camera_color_optical_frame",
        mount_parent_frame="head_end_effector_mount_link",
        camera_desc="头部双目左彩色相机（等距鱼眼投影 kannala_brandt）",
    ),
    CameraSpec(
        feature_key="observation.images.head_stereo_right",
        short_name="head_stereo_right",
        raw_mp4_name="camera_head_right_color.mp4",
        image_topic="/front_head_camera/right_color/image_raw",
        info_topic="/front_head_camera/right_color/camera_info",
        optical_frame="head_right_camera_color_optical_frame",
        mount_parent_frame="head_left_camera_color_optical_frame",
        camera_desc="头部双目右彩色相机（等距鱼眼投影 kannala_brandt）",
    ),
    CameraSpec(
        feature_key="observation.images.hand_left",
        short_name="hand_left",
        raw_mp4_name="camera_hand_left_color.mp4",
        image_topic="/left_arm_camera/color/image_raw",
        info_topic="/left_arm_camera/color/camera_info",
        optical_frame="left_arm_camera_color_optical_frame",
        mount_parent_frame="left_arm_end_effector_mount_link",
        camera_desc="左腕部 D405 彩色相机",
    ),
    CameraSpec(
        feature_key="observation.images.hand_right",
        short_name="hand_right",
        raw_mp4_name="camera_hand_right_color.mp4",
        image_topic="/right_arm_camera/color/image_raw",
        info_topic="/right_arm_camera/color/camera_info",
        optical_frame="right_arm_camera_color_optical_frame",
        mount_parent_frame="right_arm_end_effector_mount_link",
        camera_desc="右腕部 D405 彩色相机",
    ),
)


def build_state_field_descriptions() -> List[FieldDescription]:
    """构建 observation.state（68 维）的完整字段描述契约。

    返回值:
        按索引连续排列的 FieldDescription 列表，完整覆盖 [0, 68)。
    """
    specs = [
        ("state/left_arm/joint_pos", "左臂 7 自由度关节角度向量（单位：rad，URDF 关节空间直接测量）", 7),
        ("state/right_arm/joint_pos", "右臂 7 自由度关节角度向量（单位：rad，URDF 关节空间直接测量）", 7),
        ("state/left_arm/joint_vel", "左臂 7 自由度关节角速度向量（单位：rad/s）", 7),
        ("state/right_arm/joint_vel", "右臂 7 自由度关节角速度向量（单位：rad/s）", 7),
        ("state/left_end/pos", "左臂末端安装法兰位置 [x, y, z]（单位：m，base_link 基座系右手系）", 3),
        ("state/left_end/ori", "左臂末端安装法兰姿态四元数 [qx, qy, qz, qw]（归一化，base_link 基座系）", 4),
        ("state/right_end/pos", "右臂末端安装法兰位置 [x, y, z]（单位：m，base_link 基座系右手系）", 3),
        ("state/right_end/ori", "右臂末端安装法兰姿态四元数 [qx, qy, qz, qw]（归一化，base_link 基座系）", 4),
        ("state/left_effector/position", "左夹爪行程位置（单位：m，由百分比编码器按 max_width=0.12m 换算）", 1),
        ("state/left_effector/ratio", "左夹爪开合比例 [0, 1]（0 表示全夹紧，1 表示全张开）", 1),
        ("state/left_effector/width", "左夹爪两指间距（单位：m，范围 [0, 0.12]）", 1),
        ("state/right_effector/position", "右夹爪行程位置（单位：m，由百分比编码器按 max_width=0.12m 换算）", 1),
        ("state/right_effector/ratio", "右夹爪开合比例 [0, 1]（0 表示全夹紧，1 表示全张开）", 1),
        ("state/right_effector/width", "右夹爪两指间距（单位：m，范围 [0, 0.12]）", 1),
        ("state/head/joint_pos", "头部云台 2 自由度关节角度 [head_joint1, head_joint2]（单位：rad）", 2),
        ("state/head/joint_vel", "头部云台 2 自由度关节角速度（单位：rad/s）", 2),
        ("state/leg/joint_pos", "躯干升降腿部 5 自由度关节角度 leg_joint1..5（单位：rad）", 5),
        ("state/leg/joint_vel", "躯干升降腿部 5 自由度关节角速度（单位：rad/s）", 5),
        ("state/chassis/odom_pos", "移动底盘里程计平面位置 [x, y]（单位：m，odom/base_link）", 2),
        ("state/chassis/odom_ori", "移动底盘里程计姿态四元数 [qx, qy, qz, qw]（归一化，odom/base_link）", 4),
        ("state/end/wrench", "左右臂末端六维力/力矩（未纳入本次 68 维状态向量，按规范 2.3/4.2 标注为 dimensions=0）", 0),
    ]
    result: List[FieldDescription] = []
    offset = 0
    for key, desc, dim in specs:
        indices = list(range(offset, offset + dim))
        result.append(FieldDescription(key=key, description=desc, dimensions=dim, indices=indices))
        offset += dim
    return result


def build_action_field_descriptions() -> List[FieldDescription]:
    """构建 action（68 维，与 observation.state 同构对应下一帧目标状态）的完整字段描述契约。

    返回值:
        按索引连续排列的 FieldDescription 列表，完整覆盖 [0, 68)。
    """
    state_descs = build_state_field_descriptions()
    action_descs: List[FieldDescription] = []
    for item in state_descs:
        action_key = "action/" + item.key[len("state/") :]
        action_desc = f"下一帧目标控制状态：{item.description}"
        action_descs.append(
            FieldDescription(
                key=action_key,
                description=action_desc,
                dimensions=item.dimensions,
                indices=list(item.indices),
            )
        )
    return action_descs


def get_flat_vector_names(prefix: str) -> List[str]:
    """返回 68 维状态或动作向量各维度的可读名称列表。

    参数:
        prefix: 命名前缀（'state' 或 'action'）。
    返回值:
        长度为 68 的维度名称列表。
    """
    return (
        [f"{prefix}/left_arm/joint_{i}" for i in range(1, 8)]
        + [f"{prefix}/right_arm/joint_{i}" for i in range(1, 8)]
        + [f"{prefix}/left_arm/vel_{i}" for i in range(1, 8)]
        + [f"{prefix}/right_arm/vel_{i}" for i in range(1, 8)]
        + [
            f"{prefix}/left_end/x",
            f"{prefix}/left_end/y",
            f"{prefix}/left_end/z",
            f"{prefix}/left_end/qx",
            f"{prefix}/left_end/qy",
            f"{prefix}/left_end/qz",
            f"{prefix}/left_end/qw",
            f"{prefix}/right_end/x",
            f"{prefix}/right_end/y",
            f"{prefix}/right_end/z",
            f"{prefix}/right_end/qx",
            f"{prefix}/right_end/qy",
            f"{prefix}/right_end/qz",
            f"{prefix}/right_end/qw",
            f"{prefix}/left_effector/position",
            f"{prefix}/left_effector/ratio",
            f"{prefix}/left_effector/width",
            f"{prefix}/right_effector/position",
            f"{prefix}/right_effector/ratio",
            f"{prefix}/right_effector/width",
            f"{prefix}/head/joint_1",
            f"{prefix}/head/joint_2",
            f"{prefix}/head/vel_1",
            f"{prefix}/head/vel_2",
        ]
        + [f"{prefix}/leg/joint_{i}" for i in range(1, 6)]
        + [f"{prefix}/leg/vel_{i}" for i in range(1, 6)]
        + [
            f"{prefix}/chassis/odom_x",
            f"{prefix}/chassis/odom_y",
            f"{prefix}/chassis/odom_qx",
            f"{prefix}/chassis/odom_qy",
            f"{prefix}/chassis/odom_qz",
            f"{prefix}/chassis/odom_qw",
        ]
    )


def normalize_skill_name(raw_skill: str, action_text: str) -> str:
    """将原始标注中的技能词归一化到规范 10.2 节标准技能词表。

    参数:
        raw_skill: 原始 task_info.json 中的 skill 字符串。
        action_text: 动作中文描述文本，用于在 raw_skill 为空时推断技能。
    返回值:
        标准技能词表中的技能名称。
    """
    cleaned = raw_skill.strip()
    if cleaned in STANDARD_SKILL_VOCAB:
        return cleaned
    if cleaned.lower() == "grasp":
        return "Pick"
    if "抓" in action_text or "拿" in action_text:
        return "Pick"
    if "放" in action_text and "花瓶" in action_text:
        return "Lower"
    if "放置" in action_text or "放入" in action_text or "放" in action_text:
        return "Place"
    if "插" in action_text:
        return "Insert"
    return "Pick"


def compute_file_sha256(file_path: Path) -> str:
    """计算文件 SHA256 十六进制摘要。

    参数:
        file_path: 目标文件路径。
    返回值:
        SHA256 小写十六进制字符串。
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(FILE_IO_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()
