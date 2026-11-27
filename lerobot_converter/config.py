# -*- coding: utf-8 -*-
"""全量转换与验收脚本的可维护参数变量配置模块（直接在此文件统一修改维护）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class TaskMappingConfig:
    """单个原始任务目录到目标 LeRobot 数据集的映射参数配置。"""

    # 原始任务文件夹名称（位于 INPUT_ROOT 下）
    source_dir_name: str
    # 目标输出数据集目录名称（位于 OUTPUT_ROOT 下）
    target_dataset_name: str
    # 规范定义的机器人本体型号名称
    robot_type: str
    # 数据集语义版本号
    data_version: str


# ==================== 一、输入输出路径与任务列表配置 ====================

# 原始采集任务目录所在根目录
INPUT_ROOT: Path = Path(r"f:\data-sample")

# 转换后 LeRobot 数据集输出总目录
OUTPUT_ROOT: Path = Path(r"f:\data-sample\lerobot_output")

# 机器人本体 URDF 描述与 meshes 网格资源根目录
URDF_DESC_ROOT: Path = Path(r"F:\config\galbot_description\galbot_one_golf_description")

# 默认机器人本体型号与数据集版本
ROBOT_TYPE: str = "galbot_one_golf"
DATA_VERSION: str = "v1.0.0"

# 需要批量转换的任务映射清单（按顺序逐一转换）
TASK_MAPPINGS: Tuple[TaskMappingConfig, ...] = (
    TaskMappingConfig(
        source_dir_name="插花艺术",
        target_dataset_name="task_flower_arrangement",
        robot_type=ROBOT_TYPE,
        data_version=DATA_VERSION,
    ),
    TaskMappingConfig(
        source_dir_name="电线分类",
        target_dataset_name="task_wire_classification",
        robot_type=ROBOT_TYPE,
        data_version=DATA_VERSION,
    ),
)

# ==================== 二、数据源与视频编码控制开关 ====================

# 视频转换来源控制：'mp4' 表示默认读取目录现有 camera_*.mp4，'mcap' 表示从 SYNC.mcap 图像流单遍直出编码
VIDEO_SOURCE: str = "mp4"

# 非视频状态数据来源控制：'mcap' 表示优先从 SYNC.mcap 提取（缺失回退 JSON），'json' 表示读取 JSON
STATE_SOURCE: str = "mcap"

# HEVC 编码器控制：'auto' 自动探测 hevc_nvenc/hevc_amf/hevc_qsv 硬件加速并在不可用时回退 libx265
VIDEO_CODEC: str = "auto"

# 使用 libx265 软件编码时的预设档位（如 superfast / fast / medium）
X265_PRESET: str = "superfast"

# 视频质量参数（libx265 对应 CRF，硬件编码器对应恒定 QP）及关键帧参数
VIDEO_CRF_OR_QP: int = 23
VIDEO_GOP_SIZE: int = 8
VIDEO_MAX_B_FRAMES: int = 0

# ==================== 三、目标格式与硬件规格常量 ====================

# 统一交付视频分辨率（像素）与帧率（Hz）
TARGET_VIDEO_WIDTH: int = 640
TARGET_VIDEO_HEIGHT: int = 480
TARGET_VIDEO_FPS: int = 30

# Galbot G1 平行夹爪最大两指间距（米）
GALBOT_GRIPPER_MAX_WIDTH_M: float = 0.12

# LeRobot v3.0 info.json 分片上限参数
DATA_CHUNKS_SIZE: int = 1000
DATA_FILES_SIZE_IN_MB: int = 100
VIDEO_FILES_SIZE_IN_MB: int = 200

# ==================== 四、I/O 与并发性能优化参数 ====================

# mp4 模式下并发预解析各集 SYNC.mcap 的最大线程数
MCAP_PREFETCH_MAX_WORKERS: int = 3

# 单集内四路相机并发转码与统计的工作线程数
CAMERA_ENCODE_WORKERS: int = 4

# 单路 ffmpeg 转码输入解码线程数与 libx265 每路内部线程池大小
MP4_TRANSCODE_INPUT_THREADS: int = 4
X265_POOLS_PER_STREAM: int = 4
MCAP_PIPE_X265_POOLS: int = 2

# URDF 网格并发拷贝与单遍分块读写缓冲大小（字节，默认 1 MiB）
MESH_COPY_WORKERS: int = 8
FILE_IO_CHUNK_BYTES: int = 1_048_576

# 视频像素统计（关键帧抽样）每路最多采样帧数
STATS_MAX_KEYFRAMES: int = 100

# ==================== 五、独立验收脚本默认路径与阈值 ====================

# validate_lerobot.py 默认批量校验的输出根目录与可选单数据集覆盖路径
VALIDATE_ROOT_DIR: Path = OUTPUT_ROOT
VALIDATE_DATASET_DIR: Optional[Path] = None
VALIDATE_WRITE_REPORT: bool = False

# 验收阶段的并发线程池上限
VALIDATE_VIDEO_MAX_WORKERS: int = 8
VALIDATE_MESH_MAX_WORKERS: int = 8

# P00 / P0 / P1 验收校验阈值
PTS_TOLERANCE_S: float = 1e-4
FK_SAMPLE_FRAMES_PER_EP: int = 20
FK_MAX_POS_ERROR_MM: float = 5.0
FK_MAX_ROT_ERROR_DEG: float = 1.0
PROJ_SAMPLE_FRAMES_PER_EP: int = 20
MIN_PROJ_IN_VIEW_RATIO: float = 0.8
MIN_STATE_COMPLETENESS_RATIO: float = 0.99
MIN_SEGMENT_COVERAGE_RATIO: float = 0.8
MIN_SEGMENT_DURATION_S: float = 0.5
MAX_SEGMENT_DURATION_S: float = 30.0
MAX_SEGMENT_BOUNDARY_GAP_FRAMES: int = 2
MIN_EPISODE_DURATION_S: float = 10.0
MIN_VIDEO_SHORT_SIDE_PX: int = 480
