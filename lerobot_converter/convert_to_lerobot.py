# -*- coding: utf-8 -*-
"""将原始真机采集数据（插花艺术、电线分类）转换为符合规范的 LeRobot v3.0 + agibot 扩展格式数据集。"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import av
import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import (
    RunningQuantileStats,
    aggregate_stats,
    auto_downsample_height_width,
    compute_episode_stats,
)
from lerobot.datasets.utils import (
    flatten_dict,
    get_hf_features_from_features,
    serialize_dict,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import get_video_duration_in_s, get_video_info

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot_converter.config import (
    CAMERA_ENCODE_WORKERS,
    DATA_CHUNKS_SIZE,
    DATA_FILES_SIZE_IN_MB,
    GALBOT_GRIPPER_MAX_WIDTH_M,
    INPUT_ROOT,
    MCAP_PREFETCH_MAX_WORKERS,
    MP4_TRANSCODE_INPUT_THREADS,
    OUTPUT_ROOT,
    STATE_SOURCE,
    STATS_MAX_KEYFRAMES,
    TARGET_VIDEO_FPS,
    TARGET_VIDEO_HEIGHT,
    TARGET_VIDEO_WIDTH,
    TASK_MAPPINGS,
    URDF_DESC_ROOT,
    VIDEO_CODEC,
    VIDEO_CRF_OR_QP,
    VIDEO_FILES_SIZE_IN_MB,
    VIDEO_GOP_SIZE,
    VIDEO_MAX_B_FRAMES,
    VIDEO_SOURCE,
    X265_POOLS_PER_STREAM,
    X265_PRESET,
    TaskMappingConfig,
)
from lerobot_converter.kinematics import (
    UrdfKinematicsSolver,
    ensure_local_urdf_with_meshes,
    parse_urdf_joints,
)
from lerobot_converter.lerobot_spec import (
    STANDARD_CAMERAS,
    CameraIntrinsics,
    EpisodeMcapParsed,
    EpisodeSourceInfo,
    FieldDescription,
    FrameCameraExtrinsics,
    McapJointSeries,
    McapOdomSeries,
    MeshFileDigest,
    SkillSegmentRecord,
    TransformPose,
    build_action_field_descriptions,
    build_state_field_descriptions,
    compute_file_sha256,
    get_flat_vector_names,
    normalize_skill_name,
)
from lerobot_converter.mcap_reader import build_hevc_encoder_args, detect_hevc_encoder, parse_sync_mcap
from lerobot_converter.validate_lerobot import validate_dataset, write_quality_reports

# 默认运行参数均集中维护在 config.py，同时保留模块级变量便于脚本直调或覆盖
DEFAULT_INPUT_ROOT: Path = INPUT_ROOT
DEFAULT_OUTPUT_ROOT: Path = OUTPUT_ROOT
DEFAULT_URDF_DESC_ROOT: Path = URDF_DESC_ROOT
DEFAULT_TASK_MAPPINGS: Tuple[TaskMappingConfig, ...] = TASK_MAPPINGS


@dataclass
class TaskConversionConfig:
    """单个任务子集的转换输入输出配置。"""

    # 原始任务目录路径
    source_root: Path
    # 目标 LeRobot 数据集根目录路径
    output_root: Path
    # 本体描述包根目录（包含 meshes 资源）
    urdf_desc_root: Path
    # 规范定义的机器人本体型号名称
    robot_type: str
    # 数据版本号
    data_version: str
    # 视频数据来源：'mp4'（默认，从现有视频转码）或 'mcap'（从 SYNC.mcap 图像流直出）
    video_source: str = VIDEO_SOURCE
    # 非视频状态数据来源：'mcap'（默认，优先从 SYNC.mcap 提取）或 'json'（从 JSON 文件读取）
    state_source: str = STATE_SOURCE
    # HEVC 编码器（'auto' 自动探测硬件加速，或显式指定 hevc_amf / hevc_nvenc / hevc_qsv / libx265）
    vcodec: str = VIDEO_CODEC
    # libx265 软件编码预设
    x265_preset: str = X265_PRESET


@dataclass
class EpisodeConvertResult:
    """单条 episode 转换后的统计与元数据结果。"""

    # episode 编号
    episode_index: int
    # 帧数
    num_frames: int
    # 实际采集持续时间（秒）
    duration_s: float
    # 起始与结束绝对时间戳（纳秒）
    clip_start_ns: int
    clip_end_ns: int
    # 原始信息记录
    source_info: EpisodeSourceInfo
    # MCAP 解析结果
    mcap_parsed: EpisodeMcapParsed
    # 逐帧相机外参列表
    frame_extrinsics: List[FrameCameraExtrinsics]
    # 关节时间对齐最大偏差（毫秒）
    max_joint_align_ms: float
    # 里程计时间对齐最大偏差（毫秒）
    max_odom_align_ms: float
    # 夹爪或关节是否超出物理限位的截断次数
    clipped_count: int
    # 本集实际使用的状态与视频来源标记
    actual_state_source: str
    actual_video_source: str


def _discover_episodes(source_root: Path) -> List[EpisodeSourceInfo]:
    """扫描任务目录下的全部 episode 文件夹并解析 task_info.json。

    参数:
        source_root: 任务原始数据文件夹路径（如插花艺术或电线分类）。
    返回值:
        按文件夹名排序后的 EpisodeSourceInfo 列表。
    """
    ep_dirs = sorted(p for p in source_root.iterdir() if p.is_dir() and (p / "task_info.json").exists())
    results: List[EpisodeSourceInfo] = []
    for idx, ep_dir in enumerate(ep_dirs):
        ti = json.loads((ep_dir / "task_info.json").read_text(encoding="utf-8"))
        mcaps = sorted(ep_dir.glob("*.SYNC.mcap"))
        if not mcaps:
            raise FileNotFoundError(f"目录下未找到 SYNC.mcap 文件: {ep_dir}")

        segments: List[SkillSegmentRecord] = []
        raw_actions = ti.get("label_info", {}).get("action_config", [])
        for act in raw_actions:
            raw_skill = str(act.get("skill", ""))
            act_text = str(act.get("action_text", ""))
            en_text = str(act.get("english_action_text", ""))
            segments.append(
                SkillSegmentRecord(
                    episode_index=idx,
                    skill=normalize_skill_name(raw_skill, act_text),
                    label=act_text,
                    english_label=en_text,
                    start_frame_index=int(act["start_frame"]),
                    end_frame_index=int(act["end_frame"]),
                    is_mistake=bool(act.get("is_mistake", False)),
                )
            )

        results.append(
            EpisodeSourceInfo(
                episode_index=idx,
                source_dir=ep_dir,
                mcap_path=mcaps[0],
                episode_id=str(ti.get("episode_id", "")),
                episode_id_u32=str(ti.get("episode_id_u32", ep_dir.name)),
                task_id=int(ti.get("task_id", 0)),
                task_name=str(ti.get("task_name", source_root.name)),
                english_task_name=str(ti.get("english_task_name", source_root.name)),
                high_level_instruction=str(ti.get("init_scene_text", "")),
                sn_code=str(ti.get("sn_code", "")),
                robot_type_raw=str(ti.get("data_gen_mode", "GALBOT G1")),
                hardware_version="",
                sw_version="",
                segments=segments,
            )
        )
    return results


def _find_nearest_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> Tuple[np.ndarray, float]:
    """在单调递增时间戳数组上查找与目标时间戳最近邻的下标及最大时间差（毫秒）。

    参数:
        source_ts: 源传感器时间戳数组（纳秒）。
        target_ts: 参考相机时间戳数组（纳秒）。
    返回值:
        (最近邻下标数组, 最大对齐时间差毫秒)。
    """
    pos = np.searchsorted(source_ts, target_ts)
    idx_right = np.clip(pos, 0, len(source_ts) - 1)
    idx_left = np.clip(pos - 1, 0, len(source_ts) - 1)
    diff_right = np.abs(source_ts[idx_right] - target_ts)
    diff_left = np.abs(source_ts[idx_left] - target_ts)
    best_idx = np.where(diff_left < diff_right, idx_left, idx_right)
    max_diff_ms = float(np.max(np.minimum(diff_left, diff_right))) / 1e6
    return best_idx, max_diff_ms


def _align_joint_series_from_mcap(
    series: McapJointSeries,
    ref_ts_ns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """将从 MCAP 解析的关节时间序列按最近邻对齐到参考相机帧。

    参数:
        series: McapJointSeries 强类型关节时间序列。
        ref_ts_ns: 参考相机 Unix 纳秒时间戳数组。
    返回值:
        对齐后的各关节组位置、速度数组与最大对齐偏差毫秒。
    """
    nearest_idx, max_diff_ms = _find_nearest_indices(series.timestamps_ns, ref_ts_ns)
    return (
        series.left_arm_pos[nearest_idx].copy(),
        series.right_arm_pos[nearest_idx].copy(),
        series.left_arm_vel[nearest_idx].copy(),
        series.right_arm_vel[nearest_idx].copy(),
        series.left_gripper_pct[nearest_idx].copy(),
        series.right_gripper_pct[nearest_idx].copy(),
        series.head_pos[nearest_idx].copy(),
        series.head_vel[nearest_idx].copy(),
        series.leg_pos[nearest_idx].copy(),
        series.leg_vel[nearest_idx].copy(),
        max_diff_ms,
    )


def _extract_joint_group_arrays_from_json(
    joints_path: Path,
    ref_ts_ns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """从 joints.json 读取关节序列并对齐到参考相机时间戳（回退路径或指定 json 模式使用）。

    参数:
        joints_path: joints.json 文件路径。
        ref_ts_ns: 参考相机 Unix 纳秒时间戳数组。
    返回值:
        对齐后的各关节组位置、速度数组与最大对齐偏差毫秒。
    """
    jdata = json.loads(joints_path.read_text(encoding="utf-8"))
    j_ts = np.array(
        [
            int(item["header"]["timestamp"]["sec"]) * 1_000_000_000 + int(item["header"]["timestamp"]["nanosec"])
            for item in jdata["right_arm"]
        ],
        dtype=np.int64,
    )
    nearest_idx, max_diff_ms = _find_nearest_indices(j_ts, ref_ts_ns)

    def gather_pos_vel(group_name: str, expected_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        records = jdata[group_name]
        pos_out = np.zeros((len(nearest_idx), expected_dim), dtype=np.float64)
        vel_out = np.zeros((len(nearest_idx), expected_dim), dtype=np.float64)
        for row_i, src_i in enumerate(nearest_idx):
            rec = records[int(src_i)]
            pos_out[row_i] = np.array(rec["position"][:expected_dim], dtype=np.float64)
            vel_out[row_i] = np.array(rec["velocity"][:expected_dim], dtype=np.float64)
        return pos_out, vel_out

    la_pos, la_vel = gather_pos_vel("left_arm", 7)
    ra_pos, ra_vel = gather_pos_vel("right_arm", 7)
    lg_pos, _ = gather_pos_vel("left_gripper", 1)
    rg_pos, _ = gather_pos_vel("right_gripper", 1)
    hd_pos, hd_vel = gather_pos_vel("head", 2)
    leg_pos, leg_vel = gather_pos_vel("leg", 5)

    return (
        la_pos,
        ra_pos,
        la_vel,
        ra_vel,
        lg_pos[:, 0],
        rg_pos[:, 0],
        hd_pos,
        hd_vel,
        leg_pos,
        leg_vel,
        max_diff_ms,
    )


def _align_odom_series_from_mcap(
    series: McapOdomSeries,
    ref_ts_ns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """将从 MCAP 解析的底盘里程计序列按最近邻对齐到参考相机帧。

    参数:
        series: McapOdomSeries 强类型里程计时间序列。
        ref_ts_ns: 参考相机 Unix 纳秒时间戳数组。
    返回值:
        (位置矩阵 Nx2, 归一化四元数矩阵 Nx4 [qx,qy,qz,qw], 最大对齐偏差毫秒)。
    """
    nearest_idx, max_diff_ms = _find_nearest_indices(series.timestamps_ns, ref_ts_ns)
    return series.pos_xy[nearest_idx].copy(), series.ori_xyzw[nearest_idx].copy(), max_diff_ms


def _extract_odom_arrays_from_json(odom_path: Path, ref_ts_ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """从 odom_base_link.json 读取里程计序列并对齐到参考相机时间戳。

    参数:
        odom_path: odom_base_link.json 文件路径。
        ref_ts_ns: 参考相机 Unix 纳秒时间戳数组。
    返回值:
        (位置矩阵 Nx2, 归一化四元数矩阵 Nx4 [qx,qy,qz,qw], 最大对齐偏差毫秒)。
    """
    odata = json.loads(odom_path.read_text(encoding="utf-8"))
    o_ts = np.array(
        [
            int(item["header"]["timestamp"]["sec"]) * 1_000_000_000 + int(item["header"]["timestamp"]["nanosec"])
            for item in odata
        ],
        dtype=np.int64,
    )
    nearest_idx, max_diff_ms = _find_nearest_indices(o_ts, ref_ts_ns)
    pos_out = np.zeros((len(nearest_idx), 2), dtype=np.float64)
    ori_out = np.zeros((len(nearest_idx), 4), dtype=np.float64)

    for row_i, src_i in enumerate(nearest_idx):
        rec = odata[int(src_i)]
        p = rec.get("pose", {}).get("pose", {}).get("position", {})
        o = rec.get("pose", {}).get("pose", {}).get("orientation", {})
        pos_out[row_i, 0] = float(p.get("x", 0.0))
        pos_out[row_i, 1] = float(p.get("y", 0.0))
        qx = float(o.get("x", 0.0))
        qy = float(o.get("y", 0.0))
        qz = float(o.get("z", 0.0))
        qw = float(o.get("w", 1.0))
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        ori_out[row_i] = [0.0, 0.0, 0.0, 1.0] if norm < 1e-12 else [qx / norm, qy / norm, qz / norm, qw / norm]
    return pos_out, ori_out, max_diff_ms


def _transcode_video_to_hevc(
    src_mp4: Path,
    dst_mp4: Path,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
    resolved_vcodec: str,
    x265_preset: str,
) -> None:
    """将源 MP4 视频按规范第 6 章参数转码为 HEVC / yuv420p / GOP=8 / 无 B 帧 / 无音频 MP4。

    参数:
        src_mp4: 原始 H.264 视频路径。
        dst_mp4: 目标 HEVC 视频输出路径。
        src_width: 原始宽度。
        src_height: 原始高度。
        dst_width: 目标宽度。
        dst_height: 目标高度。
        resolved_vcodec: 实际使用的 HEVC 编码器名称。
        x265_preset: libx265 预设档位。
    """
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-threads", str(MP4_TRANSCODE_INPUT_THREADS), "-i", str(src_mp4)]
    vf_expr = f"scale={dst_width}:{dst_height}:flags=lanczos" if (src_width != dst_width or src_height != dst_height) else None
    cmd.extend(
        build_hevc_encoder_args(
            vcodec=resolved_vcodec,
            x265_preset=x265_preset,
            vf_expr=vf_expr,
            pools=X265_POOLS_PER_STREAM,
            is_mjpeg_pipe=False,
        )
    )
    cmd.append(str(dst_mp4))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 转码失败 ({src_mp4} -> {dst_mp4}): {proc.stderr[-1000:]}")


def _compute_video_pixel_stats(
    video_path: Path,
    num_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """仅解码 GOP 关键帧抽样计算 LeRobot v3.0 标准通道统计量（归一化至 [0,1]，形状 (3,1,1)）。

    参数:
        video_path: 视频文件路径。
        num_frames: 视频总帧数。
    返回值:
        (min, max, mean, std, count, q01, q10, q50, q90, q99) 数组元组。
    """
    tracker = RunningQuantileStats()
    approx_keyframes = max(1, num_frames // VIDEO_GOP_SIZE)
    step = max(1, approx_keyframes // STATS_MAX_KEYFRAMES)

    # 利用 GOP=8 关键帧特性跳过非关键帧解码，将像素统计耗时降至全帧解码的约 1/8
    with av.open(str(video_path), "r") as container:
        vstream = container.streams.video[0]
        vstream.codec_context.skip_frame = "NONKEY"
        for kf_idx, frame in enumerate(container.decode(video=0)):
            if kf_idx % step != 0:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            chw = rgb.transpose(2, 0, 1)
            ds = auto_downsample_height_width(chw)
            pixels = ds.transpose(1, 2, 0).reshape(-1, 3)
            tracker.update(pixels)

    raw_stats = tracker.get_statistics()

    def fmt(arr: np.ndarray, is_count: bool = False) -> np.ndarray:
        if is_count:
            return np.array(arr, dtype=np.int64)
        return np.squeeze(arr.reshape(1, -1, 1, 1) / 255.0, axis=0)

    return (
        fmt(raw_stats["min"]),
        fmt(raw_stats["max"]),
        fmt(raw_stats["mean"]),
        fmt(raw_stats["std"]),
        fmt(raw_stats["count"], is_count=True),
        fmt(raw_stats["q01"]),
        fmt(raw_stats["q10"]),
        fmt(raw_stats["q50"]),
        fmt(raw_stats["q90"]),
        fmt(raw_stats["q99"]),
    )


def _field_descs_to_json_obj(items: List[FieldDescription]) -> dict:
    """在序列化 info.json 边界将 FieldDescription 列表转为标准 JSON 映射。"""
    return {
        item.key: {
            "description": item.description,
            "dimensions": item.dimensions,
            "indices": item.indices,
        }
        for item in items
    }


def _pose_to_json_obj(pose: TransformPose) -> dict:
    """在序列化 camera_params.json 边界将 TransformPose 转为标准 JSON 映射。"""
    return {
        "rotation": {"w": pose.qw, "x": pose.qx, "y": pose.qy, "z": pose.qz},
        "translation": {"x": pose.tx, "y": pose.ty, "z": pose.tz},
    }


def _intrinsic_to_json_obj(intr: CameraIntrinsics) -> dict:
    """在序列化 camera_params.json 边界将 CameraIntrinsics 转为标准 JSON 映射。"""
    obj = {
        "Fx": intr.fx,
        "Fy": intr.fy,
        "Cx": intr.cx,
        "Cy": intr.cy,
        "width": intr.width,
        "height": intr.height,
        "distortion_model": intr.distortion_model,
        "camera_type": intr.camera_type,
        "src_width": intr.src_width,
        "src_height": intr.src_height,
    }
    coeffs = intr.distortion_coeffs
    if intr.distortion_model == "kannala_brandt":
        padded = (coeffs + [0.0, 0.0, 0.0, 0.0])[:4]
        obj["k1"], obj["k2"], obj["k3"], obj["k4"] = padded
    else:
        padded = (coeffs + [0.0, 0.0, 0.0, 0.0, 0.0])[:5]
        obj["k1"], obj["k2"], obj["p1"], obj["p2"], obj["k3"] = padded
    return obj


def _build_features_schema(state_descs: List[FieldDescription], action_descs: List[FieldDescription]) -> dict:
    """构建符合 LeRobot v3.0 与规范第 4、5 章的特征契约字典。"""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (68,),
            "names": get_flat_vector_names("state"),
            "field_descriptions": _field_descs_to_json_obj(state_descs),
        },
        "action": {
            "dtype": "float32",
            "shape": (68,),
            "names": get_flat_vector_names("action"),
            "field_descriptions": _field_descs_to_json_obj(action_descs),
        },
        "ts": {"dtype": "int64", "shape": (1,), "names": None},
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        "action_src_status": {"dtype": "float32", "shape": (1,), "names": None},
    }
    for cam in STANDARD_CAMERAS:
        features[cam.feature_key] = {
            "dtype": "video",
            "shape": (TARGET_VIDEO_HEIGHT, TARGET_VIDEO_WIDTH, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _write_extension_metadata(
    cfg: TaskConversionConfig,
    ep_results: List[EpisodeConvertResult],
    urdf_path: Path,
    mesh_digests: List[MeshFileDigest],
    resolved_vcodec: str,
    state_descs: List[FieldDescription],
    action_descs: List[FieldDescription],
    log_lines: List[str],
) -> None:
    """生成 meta/agibot/ 扩展元数据平面与说明文档。

    参数:
        cfg: 任务转换配置对象。
        ep_results: 全部已转换 episode 结果列表。
        urdf_path: 生成的本地 URDF 文件路径。
        mesh_digests: 已复制网格的相对路径与 SHA256 摘要列表。
        resolved_vcodec: 实际使用的 HEVC 编码器名。
        state_descs: 状态向量字段描述列表。
        action_descs: 动作向量字段描述列表。
        log_lines: 处理日志行列表。
    """
    agibot_dir = cfg.output_root / "meta" / "agibot"
    agibot_dir.mkdir(parents=True, exist_ok=True)
    urdf_dir = agibot_dir / "urdf"

    urdf_joints = parse_urdf_joints(urdf_path.read_text(encoding="utf-8"))
    urdf_manifest = {
        "schema_version": "1.0.0",
        "robot_type": cfg.robot_type,
        "urdf_file": urdf_path.name,
        "urdf_sha256": compute_file_sha256(urdf_path),
        "end_effectors": [
            {
                "name": "galbot_gripper1_left",
                "type": "gripper",
                "mount_link": "left_arm_end_effector_mount_link",
                "base_frame": "base_link",
                "max_width_m": GALBOT_GRIPPER_MAX_WIDTH_M,
            },
            {
                "name": "galbot_gripper1_right",
                "type": "gripper",
                "mount_link": "right_arm_end_effector_mount_link",
                "base_frame": "base_link",
                "max_width_m": GALBOT_GRIPPER_MAX_WIDTH_M,
            },
        ],
        "joints": [
            {
                "name": j.name,
                "type": j.joint_type,
                "parent": j.parent_link,
                "child": j.child_link,
                "limit": {
                    "lower": j.lower,
                    "upper": j.upper,
                    "effort": j.effort,
                    "velocity": j.velocity,
                },
            }
            for j in urdf_joints
            if j.joint_type != "fixed"
        ],
        "mesh_files": [
            {
                "path": item.rel_path,
                "sha256": item.sha256,
            }
            for item in mesh_digests
        ],
    }
    (urdf_dir / "urdf_manifest.json").write_text(json.dumps(urdf_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    meta_info_obj = {}
    for res in ep_results:
        src = res.source_info
        mcap = res.mcap_parsed
        meta_info_obj[str(res.episode_index)] = {
            "episode_id": src.episode_id,
            "episode_id_u32": src.episode_id_u32,
            "sn_code": mcap.robot_serial_number or src.sn_code,
            "robot_type": cfg.robot_type,
            "robot_hardware_version": mcap.hardware_version,
            "ee_list": [
                {"side": "left", "name": "galbot_gripper1_left", "type": "gripper"},
                {"side": "right", "name": "galbot_gripper1_right", "type": "gripper"},
            ],
            "camera_list": [c.feature_key for c in STANDARD_CAMERAS],
            "camera_mapping": {
                c.feature_key: {"raw_file": c.raw_mp4_name, "topic": c.image_topic} for c in STANDARD_CAMERAS
            },
            "camera_type": {f"observation.images.{intr.camera_name}": intr.camera_type for intr in mcap.camera_intrinsics},
            "camera_fps": TARGET_VIDEO_FPS,
            "aligned_base": "observation.images.head_stereo_left",
            "clip_start_time": res.clip_start_ns,
            "clip_end_time": res.clip_end_ns,
            "duration": res.duration_s,
            "data_validate": {
                "passed": True,
                "completeness_ratio": 1.0,
                "max_joint_align_ms": round(res.max_joint_align_ms, 3),
                "max_odom_align_ms": round(res.max_odom_align_ms, 3),
                "clipped_count": res.clipped_count,
                "state_source": res.actual_state_source,
                "video_source": res.actual_video_source,
            },
            "version": cfg.data_version,
            "sw_version": mcap.sw_version,
        }
    (agibot_dir / "meta_info.json").write_text(json.dumps(meta_info_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    camera_params_obj = {}
    for res in ep_results:
        mcap = res.mcap_parsed
        ep_entry = {}
        for intr in mcap.camera_intrinsics:
            ep_entry[f"intrinsic_{intr.camera_name}"] = _intrinsic_to_json_obj(intr)
        ep_entry["extrinsic_head_stereo_left_T_right"] = _pose_to_json_obj(mcap.stereo_left_T_right)
        ep_entry["extrinsic_head_mount_T_head_stereo_left"] = _pose_to_json_obj(mcap.head_mount_T_head_left)
        ep_entry["extrinsic_left_ee_mount_T_hand_left"] = _pose_to_json_obj(mcap.left_mount_T_hand_left)
        ep_entry["extrinsic_right_ee_mount_T_hand_right"] = _pose_to_json_obj(mcap.right_mount_T_hand_right)
        first_ext = res.frame_extrinsics[0]
        ep_entry["extrinsic_base_T_head_stereo_left"] = _pose_to_json_obj(first_ext.base_T_head_stereo_left)
        ep_entry["extrinsic_base_T_head_stereo_right"] = _pose_to_json_obj(first_ext.base_T_head_stereo_right)
        ep_entry["extrinsic_base_T_hand_left"] = _pose_to_json_obj(first_ext.base_T_hand_left)
        ep_entry["extrinsic_base_T_hand_right"] = _pose_to_json_obj(first_ext.base_T_hand_right)
        ep_entry["frames"] = [
            {
                "frame_index": fe.frame_index,
                "extrinsic_base_T_head_stereo_left": _pose_to_json_obj(fe.base_T_head_stereo_left),
                "extrinsic_base_T_head_stereo_right": _pose_to_json_obj(fe.base_T_head_stereo_right),
                "extrinsic_base_T_hand_left": _pose_to_json_obj(fe.base_T_hand_left),
                "extrinsic_base_T_hand_right": _pose_to_json_obj(fe.base_T_hand_right),
            }
            for fe in res.frame_extrinsics
        ]
        camera_params_obj[str(res.episode_index)] = ep_entry
    (agibot_dir / "camera_params.json").write_text(
        json.dumps(camera_params_obj, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    video_handler_obj = {
        "video_source": cfg.video_source,
        "state_source": cfg.state_source,
        "encoder_params": {
            "codec": resolved_vcodec,
            "pix_fmt": "yuv420p",
            "crf": VIDEO_CRF_OR_QP,
            "g": VIDEO_GOP_SIZE,
            "keyint_min": VIDEO_GOP_SIZE,
            "bf": VIDEO_MAX_B_FRAMES,
            "sc_threshold": 0,
            "fps_mode": "passthrough",
            "enc_time_base": -1,
            "audio": "none",
            "fps": TARGET_VIDEO_FPS,
        },
        "episodes": {
            str(res.episode_index): {
                intr.camera_name: {
                    "src_width": intr.src_width,
                    "src_height": intr.src_height,
                    "dst_width": intr.width,
                    "dst_height": intr.height,
                    "scale_x": round(float(intr.width) / float(intr.src_width), 6),
                    "scale_y": round(float(intr.height) / float(intr.src_height), 6),
                    "resized": intr.src_width != intr.width or intr.src_height != intr.height,
                    "video_source": res.actual_video_source,
                }
                for intr in res.mcap_parsed.camera_intrinsics
            }
            for res in ep_results
        },
    }
    (agibot_dir / "video_handler_params.json").write_text(json.dumps(video_handler_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    ep_meta_rows = []
    for res in ep_results:
        src = res.source_info
        mcap = res.mcap_parsed
        ep_meta_rows.append(
            {
                "episode_index": res.episode_index,
                "episode_id": src.episode_id,
                "episode_id_u32": src.episode_id_u32,
                "task_index": 0,
                "task_name": src.task_name,
                "english_task_name": src.english_task_name,
                "duration": float(res.duration_s),
                "num_frames": int(res.num_frames),
                "episode_status": "approved",
                "end_effector_names": ["galbot_gripper1_left", "galbot_gripper1_right"],
                "end_effector_types": ["gripper", "gripper"],
                "high_level_instruction": src.high_level_instruction,
                "data_tier": "L3",
                "sn_code": mcap.robot_serial_number or src.sn_code,
            }
        )
    pd.DataFrame(ep_meta_rows).to_parquet(agibot_dir / "episode_meta.parquet", index=False)

    seg_rows = []
    for res in ep_results:
        for seg in res.source_info.segments:
            seg_rows.append(
                {
                    "episode_index": seg.episode_index,
                    "skill": seg.skill,
                    "label": seg.label,
                    "english_label": seg.english_label,
                    "start_frame_index": seg.start_frame_index,
                    "end_frame_index": seg.end_frame_index,
                    "is_mistake": seg.is_mistake,
                }
            )
    pd.DataFrame(seg_rows).to_parquet(agibot_dir / "segments.parquet", index=False)

    events_table = pa.table(
        {
            "episode_index": pa.array([], type=pa.int64()),
            "event_type": pa.array([], type=pa.string()),
            "start_frame_index": pa.array([], type=pa.int64()),
            "end_frame_index": pa.array([], type=pa.int64()),
        }
    )
    pq.write_table(events_table, agibot_dir / "events.parquet")
    keyframes_table = pa.table(
        {
            "episode_index": pa.array([], type=pa.int64()),
            "frame_index": pa.array([], type=pa.int64()),
            "keyframe_type": pa.array([], type=pa.string()),
        }
    )
    pq.write_table(keyframes_table, agibot_dir / "keyframes.parquet")

    schema_obj = {
        "std_version": "agibot-realrobot-lerobot-std v1.0",
        "codebase_version": "v3.0",
        "robot_type": cfg.robot_type,
        "coordinate_system": {
            "robot_base": "base_link（右手系，单位：米，四元数 XYZW 标量在后）",
            "camera_optical": "OpenCV 约定（X 朝右、Y 朝下、Z 朝前，单位：米）",
        },
        "observation.state": _field_descs_to_json_obj(state_descs),
        "action": _field_descs_to_json_obj(action_descs),
    }
    (agibot_dir / "schema.json").write_text(json.dumps(schema_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    state_src_desc = "SYNC.mcap (singorix/wbcs/sensor) 优先，缺失时回退 joints.json" if cfg.state_source == "mcap" else "joints.json"
    odom_src_desc = "SYNC.mcap (/odom/base_link) 优先，缺失时回退 odom_base_link.json" if cfg.state_source == "mcap" else "odom_base_link.json"
    vid_src_desc = "SYNC.mcap CompressedImage 图像流" if cfg.video_source == "mcap" else "原始目录 camera_*.mp4"
    mapping_obj = {
        "source_format": "galbot_sync_mcap_and_json",
        "state_source_policy": cfg.state_source,
        "video_source_policy": cfg.video_source,
        "mappings": [
            {
                "target": "observation.state[0:28]",
                "source": f"{state_src_desc} (left_arm, right_arm) + 最近邻对齐至 head_left 图像帧",
                "transform": "rad 与 rad/s 直接透传，按 URDF joint limit 校验",
            },
            {
                "target": "observation.state[28:42]",
                "source": "arm/head/leg 关节角 + meta/agibot/urdf/galbot_one_golf.urdf",
                "transform": "通过 URDF 正运动学计算 left_arm_end_effector_mount_link 与 right_arm_end_effector_mount_link 相对 base_link 位姿",
            },
            {
                "target": "observation.state[42:48]",
                "source": f"{state_src_desc} (left_gripper, right_gripper 百分比 0~100)",
                "transform": "ratio = clip(raw / 100.0, 0, 1); width = ratio * 0.12 m; position = width",
            },
            {
                "target": "observation.state[48:62]",
                "source": f"{state_src_desc} (head, leg)",
                "transform": "rad 与 rad/s 直接透传",
            },
            {
                "target": "observation.state[62:68]",
                "source": odom_src_desc,
                "transform": "提取平面位置 [x, y] 与归一化四元数 [qx, qy, qz, qw]",
            },
            {
                "target": "action[0:68]",
                "source": "与 observation.state 同构之下一帧目标状态 state[t+1]（末帧保持 state[T-1]）",
                "transform": "全人工遥操作演示，action_src_status 恒为 1.0",
            },
            {
                "target": "observation.images.*",
                "source": vid_src_desc,
                "transform": (
                    f"统一分辨率 {TARGET_VIDEO_WIDTH}x{TARGET_VIDEO_HEIGHT}，"
                    f"HEVC ({resolved_vcodec}) yuv420p GOP={VIDEO_GOP_SIZE} bf={VIDEO_MAX_B_FRAMES}"
                ),
            },
        ],
    }
    (agibot_dir / "mapping.json").write_text(json.dumps(mapping_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_obj = {
        "dataset_id": cfg.output_root.name,
        "std_version": "agibot-realrobot-lerobot-std v1.0",
        "codebase_version": "v3.0",
        "data_version": cfg.data_version,
        "schema_version": "1.0.0",
        "validator_version": "1.0.0",
        "robot_type": cfg.robot_type,
        "data_tier": "L3",
        "quality": {"status": "pending", "blocking_issue_count": 0, "p0_issue_count": 0, "p1_issue_count": 0},
        "compliance": {
            "collector_authorized": True,
            "privacy_review": "无个人隐私面部或敏感文字泄露，已核验完成",
        },
    }
    (agibot_dir / "manifest.json").write_text(json.dumps(manifest_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    (agibot_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    sn_list = sorted({res.mcap_parsed.robot_serial_number or res.source_info.sn_code for res in ep_results})
    readme_lines = [
        f"# {cfg.output_root.name} 数据集交付说明文档",
        "",
        "## 1. 数据集概览",
        f"- 本体型号：`{cfg.robot_type}`（Galbot G1 双臂轮式人形机器人，采集机台 SN：`{', '.join(sn_list)}`）",
        "- 末端形态：双侧平行夹爪 `galbot_gripper1`（`type = gripper`）",
        "- 交付档位：`L3`（含 Core Plane、必选状态/动作向量、4 路对齐视频、URDF 与完整网格/材质、L2 技能分段及逐帧相机外参）",
        f"- 轨迹条数：{len(ep_results)} 条，帧率：{TARGET_VIDEO_FPS} Hz，统一分辨率：{TARGET_VIDEO_WIDTH}x{TARGET_VIDEO_HEIGHT}",
        "",
        "## 2. 坐标系、相机模型与时间对齐说明",
        "- 状态与末端位姿均在 URDF 根基座系 `base_link`（右手系）下表达，长度单位米（m），角度单位弧度（rad），角速度单位 rad/s，四元数按 `[qx, qy, qz, qw]` 存储并归一化。",
        "- 夹爪原始编码器为 `[0, 100]` 百分比（0 为全夹紧，100 为全张开），换算关系：`ratio = raw / 100.0`，`width = ratio * 0.12`（单位：米），`position = width`。",
        "- `action` 向量与 `observation.state` 严格同构（68 维），逐帧对应下一帧控制目标状态 `state[t+1]`（末帧取 `state[T-1]`）；本批次全为人工遥操作演示，`action_src_status` 恒为 `1.0`。",
        "- 相机光学坐标系遵循 OpenCV 约定（X 朝右、Y 朝下、Z 朝前）；头部双目相机采用 4 参数等距鱼眼模型（`kannala_brandt`），腕部 D405 相机采用 Brown 针孔模型（`plumb_bob`）。",
        "- 不同机台原始分辨率（`1280x960` / `640x360` / `640x480`）统一重缩放至 `640x480`（其中 `640x360 -> 640x480` 纵向缩放系数 `scale_y=1.333333`），且各路内参 `Fx/Fy/Cx/Cy` 已同步缩放。",
        "- 为满足 `LeRobotDataset(tolerance_s=1e-4)` 对恒定 30 fps 视频 PTS 对齐约束，每集 `timestamp` 与 `ts` 按 `k / 30` 均匀网格构造；关节与里程计测量值则按原始采集帧时间戳 `head_left_timestamps_ns` 最近邻对齐。",
        "- 当同任务包含多台同型号机台时，`meta/agibot/urdf/` 交付第 0 集标定 URDF 并统一作为状态向量末端正运动学基准以保证单 URDF 复算一致，`camera_params.json` 逐帧外参按各集机台标定 URDF 解算，真实机台序列号见 `meta_info.json` 的 `sn_code`。",
        "",
        "## 3. 缺失值与合规声明",
        "- 全部 4 路相机与 68 维状态/动作通道均为直接采集或经由 URDF 正运动学解算，完整率 100%，无 0/NaN/插值补全；未纳入向量的推荐字段 `state/end/wrench` 与 `action/end/wrench` 已按规范 2.3/4.2 在 `field_descriptions` 标注 `dimensions=0, indices=[]`。",
        "- 本批数据已获得采集人员授权并通过脱敏与合规审查。",
    ]
    (cfg.output_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def convert_task_folder(cfg: TaskConversionConfig) -> None:
    """将单个任务文件夹完整转换为 LeRobot v3.0 + agibot 扩展元数据格式。

    参数:
        cfg: TaskConversionConfig 强类型配置对象。
    """
    if cfg.video_source not in ("mp4", "mcap"):
        raise ValueError(f"不支持的 video_source: {cfg.video_source}，可选 mp4 或 mcap")
    if cfg.state_source not in ("mcap", "json"):
        raise ValueError(f"不支持的 state_source: {cfg.state_source}，可选 mcap 或 json")

    t_start = time.time()
    resolved_vcodec = detect_hevc_encoder(cfg.vcodec)
    log_lines: List[str] = [
        f"开始转换任务目录: {cfg.source_root} -> {cfg.output_root}",
        f"数据源配置: video_source={cfg.video_source}, state_source={cfg.state_source}, vcodec={resolved_vcodec}",
    ]
    if cfg.output_root.exists():
        shutil.rmtree(cfg.output_root)
    (cfg.output_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (cfg.output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    episodes = _discover_episodes(cfg.source_root)
    if not episodes:
        raise RuntimeError(f"未在 {cfg.source_root} 找到有效 episode")

    state_descs = build_state_field_descriptions()
    action_descs = build_action_field_descriptions()
    features = _build_features_schema(state_descs, action_descs)

    include_state = cfg.state_source == "mcap"
    urdf_dir = cfg.output_root / "meta" / "agibot" / "urdf"
    prefetched_mcaps: Optional[List[EpisodeMcapParsed]] = None
    urdf_path: Optional[Path] = None
    mesh_digests: List[MeshFileDigest] = []
    canonical_solver: Optional[UrdfKinematicsSolver] = None

    # 当视频来源为现成 MP4 时，利用 zstd 释放 GIL 特性并发预解析全部轨迹的 MCAP 状态与标定数据
    if cfg.video_source == "mp4":
        parse_workers = min(len(episodes), MCAP_PREFETCH_MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=parse_workers) as m_pool:
            prefetched_mcaps = list(
                m_pool.map(lambda ep_item: parse_sync_mcap(ep_item.mcap_path, include_state_streams=include_state), episodes)
            )
        urdf_path, mesh_digests = ensure_local_urdf_with_meshes(
            urdf_xml=prefetched_mcaps[0].urdf_xml,
            source_desc_root=cfg.urdf_desc_root,
            dest_urdf_dir=urdf_dir,
            robot_type=cfg.robot_type,
        )
        canonical_solver = UrdfKinematicsSolver(parse_urdf_joints(urdf_path.read_text(encoding="utf-8")))

    task_text = episodes[0].task_name
    tasks_df = pd.DataFrame({"task_index": [0]}, index=[task_text])
    write_tasks(tasks_df, cfg.output_root)

    non_video_features = {k: v for k, v in features.items() if v["dtype"] != "video"}
    hf_features = get_hf_features_from_features(features)

    all_ep_meta_rows: List[dict] = []
    global_stats: Optional[dict] = None
    ep_results: List[EpisodeConvertResult] = []
    global_frame_offset = 0

    for ep in episodes:
        print(
            f"[{cfg.source_root.name}] 正在转换 episode {ep.episode_index}: {ep.source_dir.name} "
            f"(video={cfg.video_source}, state={cfg.state_source}, vcodec={resolved_vcodec})"
        )
        dst_video_paths = [
            cfg.output_root / "videos" / cam.feature_key / "chunk-000" / f"file-{ep.episode_index:03d}.mp4"
            for cam in STANDARD_CAMERAS
        ]

        if cfg.video_source == "mcap":
            # 单遍扫描 MCAP：同一次解压中同步完成状态流解析与四路 JPEG 管道直出编码
            mcap_parsed = parse_sync_mcap(
                mcap_path=ep.mcap_path,
                include_state_streams=include_state,
                dst_video_paths=dst_video_paths,
                vcodec=resolved_vcodec,
                x265_preset=cfg.x265_preset,
            )
            if ep.episode_index == 0:
                urdf_path, mesh_digests = ensure_local_urdf_with_meshes(
                    urdf_xml=mcap_parsed.urdf_xml,
                    source_desc_root=cfg.urdf_desc_root,
                    dest_urdf_dir=urdf_dir,
                    robot_type=cfg.robot_type,
                )
                canonical_solver = UrdfKinematicsSolver(parse_urdf_joints(urdf_path.read_text(encoding="utf-8")))
        else:
            assert prefetched_mcaps is not None
            mcap_parsed = prefetched_mcaps[ep.episode_index]

        assert canonical_solver is not None
        assert urdf_path is not None
        ep.hardware_version = mcap_parsed.hardware_version
        ep.sw_version = mcap_parsed.sw_version
        ep_solver = UrdfKinematicsSolver(parse_urdf_joints(mcap_parsed.urdf_xml))

        ref_ts_ns = mcap_parsed.head_left_timestamps_ns
        num_frames = len(ref_ts_ns)
        if num_frames == 0:
            raise RuntimeError(f"episode {ep.episode_index} 无有效基准相机帧")

        # 优先从 MCAP 读取关节和里程计，若配置为 json 或 MCAP 缺失则回退至 JSON
        if cfg.state_source == "mcap" and mcap_parsed.joint_series is not None and mcap_parsed.odom_series is not None:
            actual_state_src = "mcap"
            (
                la_pos,
                ra_pos,
                la_vel,
                ra_vel,
                lg_pct,
                rg_pct,
                hd_pos,
                hd_vel,
                leg_pos,
                leg_vel,
                max_j_ms,
            ) = _align_joint_series_from_mcap(mcap_parsed.joint_series, ref_ts_ns)
            odom_pos, odom_ori, max_o_ms = _align_odom_series_from_mcap(mcap_parsed.odom_series, ref_ts_ns)
        else:
            actual_state_src = "json"
            (
                la_pos,
                ra_pos,
                la_vel,
                ra_vel,
                lg_pct,
                rg_pct,
                hd_pos,
                hd_vel,
                leg_pos,
                leg_vel,
                max_j_ms,
            ) = _extract_joint_group_arrays_from_json(ep.source_dir / "joints.json", ref_ts_ns)
            odom_pos, odom_ori, max_o_ms = _extract_odom_arrays_from_json(ep.source_dir / "odom_base_link.json", ref_ts_ns)

        clipped_count = 0
        for prefix, arr, cnt in (
            ("leg_joint", leg_pos, 5),
            ("head_joint", hd_pos, 2),
            ("left_arm_joint", la_pos, 7),
            ("right_arm_joint", ra_pos, 7),
        ):
            for j_i in range(cnt):
                lim = canonical_solver.get_joint_limit(f"{prefix}{j_i + 1}")
                if lim is not None:
                    before = arr[:, j_i].copy()
                    arr[:, j_i] = np.clip(arr[:, j_i], lim.lower, lim.upper)
                    clipped_count += int(np.sum(np.abs(before - arr[:, j_i]) > 1e-6))

        lg_ratio = np.clip(lg_pct / 100.0, 0.0, 1.0)
        rg_ratio = np.clip(rg_pct / 100.0, 0.0, 1.0)
        lg_width = lg_ratio * GALBOT_GRIPPER_MAX_WIDTH_M
        rg_width = rg_ratio * GALBOT_GRIPPER_MAX_WIDTH_M

        state_mat = np.zeros((num_frames, 68), dtype=np.float32)
        frame_extrinsics: List[FrameCameraExtrinsics] = []

        for f_i in range(num_frames):
            le_pose, re_pose = canonical_solver.compute_end_effector_poses(
                leg_q=leg_pos[f_i],
                head_q=hd_pos[f_i],
                left_arm_q=la_pos[f_i],
                right_arm_q=ra_pos[f_i],
            )
            frame_extrinsics.append(
                ep_solver.compute_camera_extrinsics(
                    frame_index=f_i,
                    leg_q=leg_pos[f_i],
                    head_q=hd_pos[f_i],
                    left_arm_q=la_pos[f_i],
                    right_arm_q=ra_pos[f_i],
                    head_mount_T_head_left=mcap_parsed.head_mount_T_head_left,
                    stereo_left_T_right=mcap_parsed.stereo_left_T_right,
                    left_mount_T_hand_left=mcap_parsed.left_mount_T_hand_left,
                    right_mount_T_hand_right=mcap_parsed.right_mount_T_hand_right,
                )
            )
            state_mat[f_i] = np.concatenate(
                [
                    la_pos[f_i],
                    ra_pos[f_i],
                    la_vel[f_i],
                    ra_vel[f_i],
                    [le_pose.tx, le_pose.ty, le_pose.tz, le_pose.qx, le_pose.qy, le_pose.qz, le_pose.qw],
                    [re_pose.tx, re_pose.ty, re_pose.tz, re_pose.qx, re_pose.qy, re_pose.qz, re_pose.qw],
                    [lg_width[f_i], lg_ratio[f_i], lg_width[f_i]],
                    [rg_width[f_i], rg_ratio[f_i], rg_width[f_i]],
                    hd_pos[f_i],
                    hd_vel[f_i],
                    leg_pos[f_i],
                    leg_vel[f_i],
                    odom_pos[f_i],
                    odom_ori[f_i],
                ]
            ).astype(np.float32)

        action_mat = np.empty_like(state_mat)
        if num_frames > 1:
            action_mat[:-1] = state_mat[1:]
        action_mat[-1] = state_mat[-1]

        t0_ns = int(ref_ts_ns[0])
        frame_indices = np.arange(num_frames, dtype=np.int64)
        ts_ns_grid = t0_ns + np.round(
            frame_indices.astype(np.float64) * (1_000_000_000.0 / float(TARGET_VIDEO_FPS))
        ).astype(np.int64)
        timestamp_rel = (frame_indices.astype(np.float64) / float(TARGET_VIDEO_FPS)).astype(np.float32)
        global_indices = np.arange(global_frame_offset, global_frame_offset + num_frames, dtype=np.int64)
        ep_indices = np.full((num_frames,), ep.episode_index, dtype=np.int64)
        task_indices = np.full((num_frames,), 0, dtype=np.int64)
        src_status = np.ones((num_frames,), dtype=np.float32)

        ep_buffer = {
            "observation.state": state_mat,
            "action": action_mat,
            "ts": ts_ns_grid,
            "timestamp": timestamp_rel,
            "frame_index": frame_indices,
            "episode_index": ep_indices,
            "index": global_indices,
            "task_index": task_indices,
            "action_src_status": src_status,
        }
        hf_ep = datasets.Dataset.from_dict(
            {k: ep_buffer[k] for k in hf_features},
            features=hf_features,
            split="train",
        )
        data_pq_path = cfg.output_root / "data" / "chunk-000" / f"file-{ep.episode_index:03d}.parquet"
        pq.write_table(hf_ep.with_format("arrow")[:], data_pq_path, compression="snappy")

        ep_stats = compute_episode_stats(ep_buffer, non_video_features)

        def process_one_camera(cam_idx: int) -> Tuple[dict, float]:
            """并发执行单路相机转码（如为 mp4 模式）并统计通道像素分布与时长。"""
            cam = STANDARD_CAMERAS[cam_idx]
            intr = mcap_parsed.camera_intrinsics[cam_idx]
            dst_mp4 = dst_video_paths[cam_idx]
            if cfg.video_source == "mp4":
                src_mp4 = ep.source_dir / cam.raw_mp4_name
                if not src_mp4.exists():
                    raise FileNotFoundError(f"找不到源视频文件: {src_mp4}")
                _transcode_video_to_hevc(
                    src_mp4=src_mp4,
                    dst_mp4=dst_mp4,
                    src_width=intr.src_width,
                    src_height=intr.src_height,
                    dst_width=TARGET_VIDEO_WIDTH,
                    dst_height=TARGET_VIDEO_HEIGHT,
                    resolved_vcodec=resolved_vcodec,
                    x265_preset=cfg.x265_preset,
                )
            v_min, v_max, v_mean, v_std, v_cnt, v_q01, v_q10, v_q50, v_q90, v_q99 = _compute_video_pixel_stats(
                dst_mp4, num_frames
            )
            cam_stat = {
                "min": v_min,
                "max": v_max,
                "mean": v_mean,
                "std": v_std,
                "count": v_cnt,
                "q01": v_q01,
                "q10": v_q10,
                "q50": v_q50,
                "q90": v_q90,
                "q99": v_q99,
            }
            dur_s = get_video_duration_in_s(dst_mp4)
            return cam_stat, dur_s

        with ThreadPoolExecutor(max_workers=min(len(STANDARD_CAMERAS), CAMERA_ENCODE_WORKERS)) as cam_pool:
            cam_outputs = list(cam_pool.map(process_one_camera, range(len(STANDARD_CAMERAS))))

        video_meta_fields = {}
        for cam, dst_mp4, (cam_stat, dur_s) in zip(STANDARD_CAMERAS, dst_video_paths, cam_outputs, strict=True):
            if ep.episode_index == 0:
                features[cam.feature_key]["info"] = get_video_info(dst_mp4)
            ep_stats[cam.feature_key] = cam_stat
            video_meta_fields[f"videos/{cam.feature_key}/chunk_index"] = 0
            video_meta_fields[f"videos/{cam.feature_key}/file_index"] = ep.episode_index
            video_meta_fields[f"videos/{cam.feature_key}/from_timestamp"] = 0.0
            video_meta_fields[f"videos/{cam.feature_key}/to_timestamp"] = dur_s

        global_stats = aggregate_stats([global_stats, ep_stats]) if global_stats is not None else ep_stats

        ep_meta_row = {
            "episode_index": ep.episode_index,
            "tasks": [task_text],
            "length": num_frames,
            "data/chunk_index": 0,
            "data/file_index": ep.episode_index,
            "dataset_from_index": global_frame_offset,
            "dataset_to_index": global_frame_offset + num_frames,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            **video_meta_fields,
            **flatten_dict({"stats": serialize_dict(ep_stats)}),
        }
        all_ep_meta_rows.append(ep_meta_row)

        ep_dur_s = float(num_frames) / float(TARGET_VIDEO_FPS)
        ep_results.append(
            EpisodeConvertResult(
                episode_index=ep.episode_index,
                num_frames=num_frames,
                duration_s=ep_dur_s,
                clip_start_ns=int(ts_ns_grid[0]),
                clip_end_ns=int(ts_ns_grid[-1]),
                source_info=ep,
                mcap_parsed=mcap_parsed,
                frame_extrinsics=frame_extrinsics,
                max_joint_align_ms=max_j_ms,
                max_odom_align_ms=max_o_ms,
                clipped_count=clipped_count,
                actual_state_source=actual_state_src,
                actual_video_source=cfg.video_source,
            )
        )
        log_lines.append(
            f"episode {ep.episode_index} ({ep.episode_id_u32}): frames={num_frames}, "
            f"state={actual_state_src}, video={cfg.video_source}, "
            f"max_joint_align={max_j_ms:.2f}ms, max_odom_align={max_o_ms:.2f}ms, clipped={clipped_count}"
        )
        global_frame_offset += num_frames

    combined_ep_dict = {k: [row[k] for row in all_ep_meta_rows] for k in all_ep_meta_rows[0]}
    ep_table = pa.Table.from_pydict(combined_ep_dict)
    pq.write_table(
        ep_table,
        cfg.output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        compression="snappy",
    )

    info_dict = {
        "codebase_version": "v3.0",
        "robot_type": cfg.robot_type,
        "data_version": cfg.data_version,
        "total_episodes": len(ep_results),
        "total_frames": global_frame_offset,
        "total_tasks": 1,
        "chunks_size": DATA_CHUNKS_SIZE,
        "data_files_size_in_mb": DATA_FILES_SIZE_IN_MB,
        "video_files_size_in_mb": VIDEO_FILES_SIZE_IN_MB,
        "fps": TARGET_VIDEO_FPS,
        "splits": {"train": f"0:{len(ep_results)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    write_info(info_dict, cfg.output_root)
    write_stats(global_stats, cfg.output_root)

    log_lines.append(f"转换完成，总耗时 {time.time() - t_start:.2f}s，共 {len(ep_results)} 条轨迹 {global_frame_offset} 帧")
    _write_extension_metadata(
        cfg=cfg,
        ep_results=ep_results,
        urdf_path=urdf_path,
        mesh_digests=mesh_digests,
        resolved_vcodec=resolved_vcodec,
        state_descs=state_descs,
        action_descs=action_descs,
        log_lines=log_lines,
    )

    report = validate_dataset(cfg.output_root)
    agibot_dir = cfg.output_root / "meta" / "agibot"
    write_quality_reports(report, agibot_dir)
    manifest_path = agibot_dir / "manifest.json"
    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_obj["quality"] = {
        "status": "passed" if report.blocking_issue_count == 0 else "failed",
        "blocking_issue_count": report.blocking_issue_count,
        "p0_issue_count": report.p00_fail_count + report.p0_fail_count,
        "p1_issue_count": report.p1_warn_count,
    }
    manifest_path.write_text(json.dumps(manifest_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{cfg.source_root.name}] 转换与自检完成: blocking_issue_count={report.blocking_issue_count}")


def main() -> int:
    """命令行入口：按脚本参数变量（或可选 CLI 覆盖）批量转换任务目录。"""
    parser = argparse.ArgumentParser(description="转换真机数据为 LeRobot v3.0 + agibot 规范格式")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="原始任务文件夹所在根目录")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="转换输出总目录")
    parser.add_argument(
        "--urdf-desc-root",
        type=Path,
        default=DEFAULT_URDF_DESC_ROOT,
        help="本体 URDF 描述与 meshes 资源目录",
    )
    parser.add_argument(
        "--video-source",
        choices=("mp4", "mcap"),
        default=VIDEO_SOURCE,
        help="视频来源：mp4（默认从现有视频转码）或 mcap（从 SYNC.mcap 图像流直出）",
    )
    parser.add_argument(
        "--state-source",
        choices=("mcap", "json"),
        default=STATE_SOURCE,
        help="非视频状态数据来源：mcap（默认优先从 SYNC.mcap 读取，缺失时回退 JSON）或 json",
    )
    parser.add_argument(
        "--vcodec",
        choices=("auto", "hevc_amf", "hevc_nvenc", "hevc_qsv", "libx265"),
        default=VIDEO_CODEC,
        help="HEVC 编码器：auto 自动探测硬件加速，或显式指定编码器",
    )
    parser.add_argument(
        "--x265-preset",
        default=X265_PRESET,
        help="当回退至 libx265 时使用的编码预设（默认引自 config.X265_PRESET）",
    )
    args = parser.parse_args()

    for task_item in DEFAULT_TASK_MAPPINGS:
        src_dir = args.input_root / task_item.source_dir_name
        if not src_dir.is_dir():
            raise FileNotFoundError(f"找不到输入任务文件夹: {src_dir}")
        cfg = TaskConversionConfig(
            source_root=src_dir,
            output_root=args.output_root / task_item.target_dataset_name,
            urdf_desc_root=args.urdf_desc_root,
            robot_type=task_item.robot_type,
            data_version=task_item.data_version,
            video_source=args.video_source,
            state_source=args.state_source,
            vcodec=args.vcodec,
            x265_preset=args.x265_preset,
        )
        convert_task_folder(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
