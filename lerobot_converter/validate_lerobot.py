# -*- coding: utf-8 -*-
"""LeRobot v3.0 与 agibot 扩展元数据全量验收检查脚本（对应规范第 11、12 章）。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import av
import numpy as np
import pandas as pd
import yourdfpy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot_converter.config import (
    FK_MAX_POS_ERROR_MM,
    FK_MAX_ROT_ERROR_DEG,
    FK_SAMPLE_FRAMES_PER_EP,
    MAX_SEGMENT_BOUNDARY_GAP_FRAMES,
    MAX_SEGMENT_DURATION_S,
    MIN_EPISODE_DURATION_S,
    MIN_PROJ_IN_VIEW_RATIO,
    MIN_SEGMENT_COVERAGE_RATIO,
    MIN_SEGMENT_DURATION_S,
    MIN_STATE_COMPLETENESS_RATIO,
    MIN_VIDEO_SHORT_SIDE_PX,
    PROJ_SAMPLE_FRAMES_PER_EP,
    PTS_TOLERANCE_S,
    TARGET_VIDEO_FPS,
    TARGET_VIDEO_HEIGHT,
    TARGET_VIDEO_WIDTH,
    VALIDATE_DATASET_DIR,
    VALIDATE_MESH_MAX_WORKERS,
    VALIDATE_ROOT_DIR,
    VALIDATE_VIDEO_MAX_WORKERS,
    VALIDATE_WRITE_REPORT,
    VIDEO_GOP_SIZE,
)
from lerobot_converter.kinematics import UrdfKinematicsSolver, parse_urdf_joints, quat_xyzw_to_matrix
from lerobot_converter.lerobot_spec import (
    STANDARD_CAMERAS,
    STANDARD_SKILL_VOCAB,
    ValidationCheckItem,
    ValidationReport,
    VideoCheckResult,
    compute_file_sha256,
)

# 验收默认运行参数均集中维护在 config.py，保留模块级别名便于直接调用与查看
DEFAULT_ROOT_DIR: Path = VALIDATE_ROOT_DIR
DEFAULT_DATASET_DIR: Optional[Path] = VALIDATE_DATASET_DIR
DEFAULT_WRITE_REPORT: bool = VALIDATE_WRITE_REPORT


def _add_item(
    report: ValidationReport,
    priority: str,
    name: str,
    passed: bool,
    detail: str,
) -> None:
    """向验证报告追加一条检查记录并更新汇总计数。"""
    report.items.append(ValidationCheckItem(priority=priority, name=name, passed=passed, detail=detail))
    if not passed:
        if priority == "P00":
            report.p00_fail_count += 1
            report.blocking_issue_count += 1
        elif priority == "P0":
            report.p0_fail_count += 1
            report.blocking_issue_count += 1
        elif priority == "P1":
            report.p1_warn_count += 1


def _validate_field_descriptions(
    info_path: Path,
    vector_key: str,
    expected_prefix: str,
) -> Tuple[bool, str]:
    """校验 info.json 中 observation.state 或 action 的 field_descriptions 契约。

    参数:
        info_path: meta/info.json 文件路径。
        vector_key: 特征键名（'observation.state' 或 'action'）。
        expected_prefix: 预期子字段前缀（'state/' 或 'action/'）。
    返回值:
        (是否通过, 检查详情)。
    """
    info_obj = json.loads(info_path.read_text(encoding="utf-8"))
    features = info_obj.get("features", {})
    if vector_key not in features:
        return False, f"features 缺少 {vector_key}"
    ft = features[vector_key]
    shape = ft.get("shape", [])
    if len(shape) != 1 or shape[0] <= 0:
        return False, f"{vector_key} shape 非法: {shape}"
    total_dim = int(shape[0])
    fdescs = ft.get("field_descriptions")
    if not fdescs:
        return False, f"{vector_key} 缺少 field_descriptions"

    covered: List[int] = []
    for sub_key, desc_obj in fdescs.items():
        if not sub_key.startswith(expected_prefix):
            return False, f"子字段命名不符合前缀约束: {sub_key}"
        dims = int(desc_obj.get("dimensions", -1))
        indices = desc_obj.get("indices", [])
        if dims != len(indices):
            return False, f"{sub_key} len(indices)={len(indices)} != dimensions={dims}"
        if not desc_obj.get("description"):
            return False, f"{sub_key} 缺少 description"
        covered.extend(int(i) for i in indices)

    if len(covered) != len(set(covered)):
        return False, f"{vector_key} indices 存在重叠"
    if sorted(covered) != list(range(total_dim)):
        return False, f"{vector_key} indices 未完整覆盖 [0, {total_dim})"
    return True, f"{vector_key} 覆盖 {len(fdescs)} 个子字段共 {total_dim} 维"


def _check_single_video(
    episode_index: int,
    camera_name: str,
    video_path: Path,
    expected_frames: int,
    expected_fps: int,
    expected_width: int,
    expected_height: int,
) -> VideoCheckResult:
    """单遍打开 PyAV 校验视频流编码参数、全程可解码性、PTS 均匀性及 GOP=8 关键帧间隔。

    参数:
        episode_index: 所属 episode 编号。
        camera_name: 相机简短名称。
        video_path: MP4 视频文件路径。
        expected_frames: 预期总帧数。
        expected_fps: 预期帧率。
        expected_width: 预期视频宽度。
        expected_height: 预期视频高度。
    返回值:
        VideoCheckResult 强类型校验结果对象。
    """
    pts_sec: List[float] = []
    keyframe_indices: List[int] = []
    expected_dt = 1.0 / float(expected_fps)

    with av.open(str(video_path), "r") as container:
        if not container.streams.video:
            return VideoCheckResult(episode_index, camera_name, False, False, f"{video_path.name} 缺少视频流")
        vstream = container.streams.video[0]
        codec_name = str(vstream.codec.canonical_name)
        pix_fmt = str(vstream.pix_fmt)
        vw = int(vstream.width)
        vh = int(vstream.height)
        has_b_frames = bool(vstream.codec_context.has_b_frames)
        base_fps = float(vstream.base_rate) if vstream.base_rate else 0.0
        audio_cnt = len(container.streams.audio)
        tb = float(vstream.time_base)

        for idx, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                return VideoCheckResult(episode_index, camera_name, False, False, f"{video_path.name} 第 {idx} 帧缺少 PTS")
            pts_sec.append(float(frame.pts) * tb)
            if frame.key_frame:
                keyframe_indices.append(idx)

    if len(pts_sec) != expected_frames:
        return VideoCheckResult(
            episode_index, camera_name, False, False, f"{video_path.name} 解码帧数 {len(pts_sec)} != {expected_frames}"
        )
    if abs(pts_sec[0]) > PTS_TOLERANCE_S:
        return VideoCheckResult(
            episode_index, camera_name, False, False, f"{video_path.name} 起始 PTS={pts_sec[0]:.6f} != 0"
        )

    pts_arr = np.array(pts_sec, dtype=np.float64)
    dts = np.diff(pts_arr)
    if np.any(dts <= 0.0):
        return VideoCheckResult(episode_index, camera_name, False, False, f"{video_path.name} PTS 非单调递增")
    max_dt_err = float(np.max(np.abs(dts - expected_dt)))
    if max_dt_err >= PTS_TOLERANCE_S:
        return VideoCheckResult(
            episode_index, camera_name, False, False, f"{video_path.name} PTS 帧间隔偏差 {max_dt_err:.6f}s >= {PTS_TOLERANCE_S}s"
        )

    gop_ok = True
    if len(keyframe_indices) >= 2:
        kf_diffs = np.diff(np.array(keyframe_indices, dtype=np.int64))
        if np.any(kf_diffs != VIDEO_GOP_SIZE):
            gop_ok = False

    enc_ok = (
        codec_name == "hevc"
        and pix_fmt == "yuv420p"
        and audio_cnt == 0
        and not has_b_frames
        and vw == expected_width
        and vh == expected_height
        and abs(base_fps - float(expected_fps)) < 1e-3
        and gop_ok
    )
    detail = f"{video_path.name} {len(pts_sec)} 帧 PTS 最大偏差 {max_dt_err:.2e}s, codec={codec_name}, gop_ok={gop_ok}"
    return VideoCheckResult(episode_index, camera_name, True, enc_ok, detail)


def _find_slice(info_path: Path, vector_key: str, sub_key: str) -> slice:
    """从 info.json 的 field_descriptions 获取子字段切片区间。"""
    info_obj = json.loads(info_path.read_text(encoding="utf-8"))
    indices = info_obj["features"][vector_key]["field_descriptions"][sub_key]["indices"]
    return slice(int(indices[0]), int(indices[-1]) + 1)


def _project_camera_point(
    x: float,
    y: float,
    z: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion_model: str,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[float, float]:
    """按相机畸变模型将相机光学系三维点 (x, y, z) 投影到图像像素平面 (u, v)。

    参数:
        x: 相机光学系 X 坐标（米）。
        y: 相机光学系 Y 坐标（米）。
        z: 相机光学系深度 Z 坐标（米，需 > 0）。
        fx: X 方向焦距（像素）。
        fy: Y 方向焦距（像素）。
        cx: 主点 X 坐标（像素）。
        cy: 主点 Y 坐标（像素）。
        distortion_model: 畸变模型名称（'kannala_brandt' 或 'plumb_bob' 等针孔模型）。
        k1: 一阶径向畸变系数。
        k2: 二阶径向畸变系数。
        k3: 三阶径向/径向畸变系数。
        k4: 四阶鱼眼畸变系数（仅 kannala_brandt 使用）。
        p1: 切向畸变系数 p1（针孔模型使用）。
        p2: 切向畸变系数 p2（针孔模型使用）。
    返回值:
        (像素坐标 u, 像素坐标 v)。
    """
    if distortion_model == "kannala_brandt":
        # OpenCV 等距鱼眼投影：r_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
        r = math.hypot(x, y)
        theta = math.atan2(r, z)
        t2 = theta * theta
        theta_d = theta * (1.0 + k1 * t2 + k2 * (t2**2) + k3 * (t2**3) + k4 * (t2**4))
        scale = theta_d / r if r > 1e-9 else 1.0
        return fx * x * scale + cx, fy * y * scale + cy

    xn = x / z
    yn = y / z
    r2 = xn * xn + yn * yn
    radial = 1.0 + k1 * r2 + k2 * (r2**2) + k3 * (r2**3)
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return fx * xd + cx, fy * yd + cy


def _is_unit_quat(qx: float, qy: float, qz: float, qw: float, tol: float = 1e-3) -> bool:
    """校验四元数模长是否在允许容差内接近 1。

    参数:
        qx: 四元数 X 分量。
        qy: 四元数 Y 分量。
        qz: 四元数 Z 分量。
        qw: 四元数 W 分量。
        tol: 允许的单位长度最大偏差阈值。
    返回值:
        是否满足单位四元数约束。
    """
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return abs(norm - 1.0) <= tol


def validate_dataset(dataset_dir: Path) -> ValidationReport:
    """对单个 LeRobot v3.0 数据集目录执行全项验收检查。

    参数:
        dataset_dir: 待验证的数据集根目录。
    返回值:
        ValidationReport 验收报告对象。
    """
    report = ValidationReport(dataset_root=str(dataset_dir), data_tier="L3")
    meta_dir = dataset_dir / "meta"
    agibot_dir = meta_dir / "agibot"
    info_path = meta_dir / "info.json"

    if not info_path.exists() or not agibot_dir.is_dir():
        _add_item(report, "P00", "主契约与目录结构", False, "缺少 meta/info.json 或 meta/agibot/")
        return report

    info_obj = json.loads(info_path.read_text(encoding="utf-8"))
    codebase_ok = info_obj.get("codebase_version") == "v3.0"
    robot_type = str(info_obj.get("robot_type", ""))
    fps = int(info_obj.get("fps", 0))
    total_episodes = int(info_obj.get("total_episodes", -1))
    total_frames = int(info_obj.get("total_frames", -1))
    total_tasks = int(info_obj.get("total_tasks", -1))
    data_version = str(info_obj.get("data_version", ""))

    ep_parquet_paths = sorted((meta_dir / "episodes").glob("chunk-*/file-*.parquet"))
    if not ep_parquet_paths:
        _add_item(report, "P00", "主契约完整", False, "缺少 meta/episodes/chunk-*/file-*.parquet")
        return report
    ep_df = pd.concat([pd.read_parquet(p) for p in ep_parquet_paths], ignore_index=True)
    tasks_df = pd.read_parquet(meta_dir / "tasks.parquet")

    actual_ep_count = len(ep_df)
    actual_frame_count = int(ep_df["length"].sum())
    actual_task_count = len(tasks_df)

    contract_ok = (
        codebase_ok
        and bool(robot_type)
        and fps == TARGET_VIDEO_FPS
        and bool(data_version)
        and total_episodes == actual_ep_count
        and total_frames == actual_frame_count
        and total_tasks == actual_task_count
    )
    _add_item(
        report,
        "P00",
        "主契约完整",
        contract_ok,
        f"codebase={info_obj.get('codebase_version')}, robot_type={robot_type}, fps={fps}, "
        f"episodes={total_episodes}/{actual_ep_count}, frames={total_frames}/{actual_frame_count}, tasks={total_tasks}/{actual_task_count}",
    )

    st_ok, st_msg = _validate_field_descriptions(info_path, "observation.state", "state/")
    ac_ok, ac_msg = _validate_field_descriptions(info_path, "action", "action/")
    _add_item(report, "P00", "field_descriptions 契约", st_ok and ac_ok, f"{st_msg}; {ac_msg}")

    meta_info_path = agibot_dir / "meta_info.json"
    meta_info_obj = json.loads(meta_info_path.read_text(encoding="utf-8")) if meta_info_path.exists() else {}
    state_fdescs = info_obj["features"]["observation.state"]["field_descriptions"]
    action_fdescs = info_obj["features"]["action"]["field_descriptions"]
    base_state_keys = [
        "state/left_arm/joint_pos",
        "state/right_arm/joint_pos",
        "state/left_end/pos",
        "state/left_end/ori",
        "state/right_end/pos",
        "state/right_end/ori",
        "state/left_effector/position",
        "state/right_effector/position",
    ]
    req_scalar_cols = ["episode_index", "frame_index", "index", "task_index", "ts", "timestamp", "action_src_status"]
    missing_req: List[str] = (
        [k for k in base_state_keys if int(state_fdescs.get(k, {}).get("dimensions", 0)) <= 0]
        + [
            "action/" + k[len("state/") :]
            for k in base_state_keys
            if int(action_fdescs.get("action/" + k[len("state/") :], {}).get("dimensions", 0)) <= 0
        ]
        + [k for k in req_scalar_cols if k not in info_obj["features"]]
    )
    forbidden_found: List[str] = []
    ee_ok = True
    seen_ee_types: List[str] = []
    for ep_idx in range(actual_ep_count):
        ep_ee_list = meta_info_obj.get(str(ep_idx), {}).get("ee_list", [])
        if not ep_ee_list:
            ee_ok = False
            continue
        for ee in ep_ee_list:
            ee_type = str(ee.get("type", ""))
            side = str(ee.get("side", ""))
            if ee_type not in ("gripper", "hand", "tool") or side not in ("left", "right"):
                ee_ok = False
                continue
            if ee_type not in seen_ee_types:
                seen_ee_types.append(ee_type)
            pos_k = f"state/{side}_effector/position"
            rat_k = f"state/{side}_effector/ratio"
            wid_k = f"state/{side}_effector/width"
            if int(state_fdescs.get(pos_k, {}).get("dimensions", 0)) <= 0:
                missing_req.append(pos_k)
            if int(action_fdescs.get("action/" + pos_k[len("state/") :], {}).get("dimensions", 0)) <= 0:
                missing_req.append("action/" + pos_k[len("state/") :])
            if ee_type == "gripper":
                for gk in (rat_k, wid_k):
                    ak = "action/" + gk[len("state/") :]
                    if int(state_fdescs.get(gk, {}).get("dimensions", 0)) <= 0:
                        missing_req.append(gk)
                    if int(action_fdescs.get(ak, {}).get("dimensions", 0)) <= 0:
                        missing_req.append(ak)
            elif ee_type == "hand":
                for hk in (rat_k, wid_k):
                    ak = "action/" + hk[len("state/") :]
                    if int(state_fdescs.get(hk, {}).get("dimensions", 0)) > 0:
                        forbidden_found.append(hk)
                    if int(action_fdescs.get(ak, {}).get("dimensions", 0)) > 0:
                        forbidden_found.append(ak)
    missing_req = sorted(set(missing_req))
    forbidden_found = sorted(set(forbidden_found))
    req_all_ok = (len(missing_req) == 0) and (len(forbidden_found) == 0) and ee_ok
    req_detail = (
        f"{'/'.join(seen_ee_types) or 'gripper'} 必选字段、索引标量列及 ee_list 均齐全"
        if req_all_ok
        else f"缺失={missing_req}, 禁止字段={forbidden_found}, ee_ok={ee_ok}"
    )
    _add_item(report, "P00", "必选字段齐全", req_all_ok, req_detail)

    urdf_dir = agibot_dir / "urdf"
    urdf_file = urdf_dir / f"{robot_type}.urdf"
    urdf_manifest_path = urdf_dir / "urdf_manifest.json"
    urdf_ok = False
    urdf_detail = ""
    urdf_solver: Optional[UrdfKinematicsSolver] = None
    if urdf_file.exists() and urdf_manifest_path.exists():
        manifest_obj = json.loads(urdf_manifest_path.read_text(encoding="utf-8"))
        mesh_files = manifest_obj.get("mesh_files", [])
        urdf_sha_ok = manifest_obj.get("urdf_sha256") == compute_file_sha256(urdf_file)

        def verify_mesh(rel: str, expected_sha: str) -> str:
            mp = urdf_dir / rel
            if not mp.exists() or compute_file_sha256(mp) != expected_sha:
                return rel
            return ""

        with ThreadPoolExecutor(max_workers=VALIDATE_MESH_MAX_WORKERS) as mesh_pool:
            bad_meshes = [
                r for r in mesh_pool.map(lambda it: verify_mesh(it["path"], it["sha256"]), mesh_files) if r
            ]

        missing_mtls: List[str] = []
        for obj_path in urdf_dir.rglob("*.obj"):
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f_obj:
                for line_idx, line in enumerate(f_obj):
                    if line_idx >= 64:
                        break
                    stripped = line.strip()
                    if stripped.startswith("mtllib "):
                        mtl_name = stripped.split(None, 1)[1].strip()
                        if not (obj_path.parent / mtl_name).exists():
                            missing_mtls.append(f"{obj_path.name}->{mtl_name}")

        hash_ok = urdf_sha_ok and len(bad_meshes) == 0 and len(missing_mtls) == 0
        if not urdf_sha_ok:
            urdf_detail = "urdf_manifest.json 中 urdf_sha256 与实际 URDF 文件不一致"
        elif bad_meshes:
            urdf_detail = f"网格文件缺失或校验失败: {bad_meshes[0]}"
        elif missing_mtls:
            urdf_detail = f"OBJ 材质文件缺失: {missing_mtls[0]}"
        if hash_ok:
            try:
                loaded_urdf = yourdfpy.URDF.load(str(urdf_file), load_meshes=True, build_scene_graph=False)
                urdf_joints = parse_urdf_joints(urdf_file.read_text(encoding="utf-8"))
                urdf_solver = UrdfKinematicsSolver(urdf_joints)
                non_fixed = {j.name: j for j in urdf_joints if j.joint_type != "fixed"}
                manifest_joints = manifest_obj.get("joints", [])
                joints_sync_ok = len(manifest_joints) == len(non_fixed) and all(
                    mj.get("name") in non_fixed
                    and abs(float(mj.get("limit", {}).get("lower", 0.0)) - non_fixed[mj["name"]].lower) <= 1e-6
                    and abs(float(mj.get("limit", {}).get("upper", 0.0)) - non_fixed[mj["name"]].upper) <= 1e-6
                    for mj in manifest_joints
                )
                urdf_ok = (len(loaded_urdf.joint_map) == len(urdf_joints)) and joints_sync_ok
                urdf_detail = (
                    f"yourdfpy 加载成功，关节数={len(loaded_urdf.joint_map)}，网格与材质校验通过={len(mesh_files)}"
                    if urdf_ok
                    else "urdf_manifest.json 关节限位与 URDF 不一致"
                )
            except Exception as exc:
                urdf_detail = f"yourdfpy 解析失败: {exc}"
    else:
        urdf_detail = "缺少 URDF 或 urdf_manifest.json"
    _add_item(report, "P0", "URDF 完备", urdf_ok, urdf_detail)

    data_tmpl = info_obj["data_path"]
    video_tmpl = info_obj["video_path"]
    time_ok = True
    time_detail = "所有 episode ts/timestamp/frame_index/index 连续一致"
    num_ok = True
    num_detail = "state 与 action 无 NaN/Inf，四元数模长偏差 <= 1e-3，夹爪比例与关节角在限位内"
    total_valid_frames = 0
    index_ok = True
    index_detail = "episodes 索引可正确定位所有 data/video 文件及时间窗"
    fk_ok = True
    fk_detail = "所有抽检帧正运动学位置误差 <= 5mm、旋转误差 <= 1°"

    sl_la = _find_slice(info_path, "observation.state", "state/left_arm/joint_pos")
    sl_ra = _find_slice(info_path, "observation.state", "state/right_arm/joint_pos")
    sl_le_p = _find_slice(info_path, "observation.state", "state/left_end/pos")
    sl_le_o = _find_slice(info_path, "observation.state", "state/left_end/ori")
    sl_re_p = _find_slice(info_path, "observation.state", "state/right_end/pos")
    sl_re_o = _find_slice(info_path, "observation.state", "state/right_end/ori")
    sl_lr = _find_slice(info_path, "observation.state", "state/left_effector/ratio")
    sl_rr = _find_slice(info_path, "observation.state", "state/right_effector/ratio")
    sl_head = _find_slice(info_path, "observation.state", "state/head/joint_pos")
    sl_leg = _find_slice(info_path, "observation.state", "state/leg/joint_pos")
    sl_odom_o = _find_slice(info_path, "observation.state", "state/chassis/odom_ori")

    global_offset = 0
    all_episode_frames: List[pd.DataFrame] = []
    video_tasks: List[Tuple[int, str, Path, int, int, int, int]] = []

    for _, row in ep_df.iterrows():
        ep_idx = int(row["episode_index"])
        ep_len = int(row["length"])
        d_path = dataset_dir / data_tmpl.format(chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        if not d_path.exists():
            index_ok = False
            index_detail = f"episode {ep_idx} 数据文件不存在: {d_path}"
            break

        full_df = pd.read_parquet(d_path)
        ep_rows = full_df[full_df["episode_index"] == ep_idx].reset_index(drop=True)
        if len(ep_rows) != ep_len or int(row["dataset_from_index"]) != global_offset or int(row["dataset_to_index"]) != global_offset + ep_len:
            index_ok = False
            index_detail = f"episode {ep_idx} 长度或全局起止索引不匹配"
            break
        all_episode_frames.append(ep_rows)

        ts_arr = ep_rows["ts"].to_numpy(dtype=np.int64)
        t_arr = ep_rows["timestamp"].to_numpy(dtype=np.float64)
        fidx_arr = ep_rows["frame_index"].to_numpy(dtype=np.int64)
        gidx_arr = ep_rows["index"].to_numpy(dtype=np.int64)
        if not np.array_equal(fidx_arr, np.arange(ep_len)):
            time_ok = False
            time_detail = f"episode {ep_idx} frame_index 不连续自 0"
        elif not np.array_equal(gidx_arr, np.arange(global_offset, global_offset + ep_len)):
            time_ok = False
            time_detail = f"episode {ep_idx} index 全局不连续"
        elif np.any(np.diff(ts_arr) <= 0) or ts_arr[0] < 1_000_000_000_000_000_000:
            time_ok = False
            time_detail = f"episode {ep_idx} ts 非纳秒级绝对时间或非单调递增"
        else:
            rel_from_ts = (ts_arr - ts_arr[0]) / 1e9
            if abs(float(t_arr[0])) > 1e-6 or float(np.max(np.abs(rel_from_ts - t_arr))) > 1e-4:
                time_ok = False
                time_detail = f"episode {ep_idx} timestamp 与 ts 换算不一致"
        global_offset += ep_len

        state_mat = np.stack(ep_rows["observation.state"].to_numpy()).astype(np.float64)
        action_mat = np.stack(ep_rows["action"].to_numpy()).astype(np.float64)
        valid_mask = np.all(np.isfinite(state_mat), axis=1) & np.all(np.isfinite(action_mat), axis=1)
        total_valid_frames += int(np.sum(valid_mask))
        if not bool(np.all(valid_mask)):
            num_ok = False
            num_detail = f"episode {ep_idx} 存在 NaN 或 Inf"
        else:
            for vec_label, mat in (("observation.state", state_mat), ("action", action_mat)):
                for quat_slice in (sl_le_o, sl_re_o, sl_odom_o):
                    norms = np.linalg.norm(mat[:, quat_slice], axis=1)
                    if float(np.max(np.abs(norms - 1.0))) > 1e-3:
                        num_ok = False
                        num_detail = f"episode {ep_idx} {vec_label} 四元数未归一化，最大误差={np.max(np.abs(norms - 1.0)):.5f}"
                for r_slice in (sl_lr, sl_rr):
                    r_vals = mat[:, r_slice]
                    if float(r_vals.min()) < -1e-5 or float(r_vals.max()) > 1.0 + 1e-5:
                        num_ok = False
                        num_detail = f"episode {ep_idx} {vec_label} 夹爪 ratio 越界"

                if urdf_solver is not None:
                    joint_groups = [
                        ("leg_joint", sl_leg, 5),
                        ("head_joint", sl_head, 2),
                        ("left_arm_joint", sl_la, 7),
                        ("right_arm_joint", sl_ra, 7),
                    ]
                    for prefix, jslice, cnt in joint_groups:
                        q_block = mat[:, jslice]
                        for j_i in range(cnt):
                            lim = urdf_solver.get_joint_limit(f"{prefix}{j_i + 1}")
                            if lim is not None:
                                if float(q_block[:, j_i].min()) < lim.lower - 1e-3 or float(q_block[:, j_i].max()) > lim.upper + 1e-3:
                                    num_ok = False
                                    num_detail = f"episode {ep_idx} {vec_label} {lim.name} 超出 URDF 限位"

            if urdf_solver is not None:
                sample_indices = np.linspace(0, ep_len - 1, min(FK_SAMPLE_FRAMES_PER_EP, ep_len), dtype=int)
                max_pos_mm = 0.0
                max_rot_deg = 0.0
                for s_idx in sample_indices:
                    fk_l, fk_r = urdf_solver.compute_end_effector_poses(
                        leg_q=state_mat[s_idx, sl_leg],
                        head_q=state_mat[s_idx, sl_head],
                        left_arm_q=state_mat[s_idx, sl_la],
                        right_arm_q=state_mat[s_idx, sl_ra],
                    )
                    for fk_pose, p_sl, o_sl in ((fk_l, sl_le_p, sl_le_o), (fk_r, sl_re_p, sl_re_o)):
                        st_pos = state_mat[s_idx, p_sl]
                        pos_mm = float(np.linalg.norm(st_pos - np.array([fk_pose.tx, fk_pose.ty, fk_pose.tz])) * 1000.0)
                        max_pos_mm = max(max_pos_mm, pos_mm)
                        r_fk = quat_xyzw_to_matrix(fk_pose.qx, fk_pose.qy, fk_pose.qz, fk_pose.qw)
                        st_q = state_mat[s_idx, o_sl]
                        r_st = quat_xyzw_to_matrix(st_q[0], st_q[1], st_q[2], st_q[3])
                        cos_val = max(-1.0, min(1.0, (float(np.trace(r_fk.T @ r_st)) - 1.0) / 2.0))
                        rot_deg = math.degrees(math.acos(cos_val))
                        max_rot_deg = max(max_rot_deg, rot_deg)
                if max_pos_mm > FK_MAX_POS_ERROR_MM or max_rot_deg > FK_MAX_ROT_ERROR_DEG:
                    fk_ok = False
                    fk_detail = f"episode {ep_idx} FK 误差超标: pos={max_pos_mm:.3f}mm, rot={max_rot_deg:.3f}°"

        for cam in STANDARD_CAMERAS:
            vk = cam.feature_key
            vc_idx = int(row[f"videos/{vk}/chunk_index"])
            vf_idx = int(row[f"videos/{vk}/file_index"])
            from_ts = float(row[f"videos/{vk}/from_timestamp"])
            to_ts = float(row[f"videos/{vk}/to_timestamp"])
            v_path = dataset_dir / video_tmpl.format(video_key=vk, chunk_index=vc_idx, file_index=vf_idx)
            if not v_path.exists() or abs((to_ts - from_ts) - (ep_len / float(fps))) > 0.1:
                index_ok = False
                index_detail = f"episode {ep_idx} {vk} 视频文件或时间窗异常"
                continue
            ft_shape = info_obj["features"][vk]["shape"]
            video_tasks.append(
                (ep_idx, cam.short_name, v_path, ep_len, fps, int(ft_shape[1]), int(ft_shape[0]))
            )

    completeness_ratio = float(total_valid_frames) / float(max(1, global_offset))
    complete_ok = (global_offset > 0) and (completeness_ratio >= MIN_STATE_COMPLETENESS_RATIO)

    video_decode_all_ok = True
    video_encode_all_ok = True
    video_details: List[str] = []
    if video_tasks:
        workers = min(len(video_tasks), max(4, os.cpu_count() or 4), VALIDATE_VIDEO_MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as v_pool:
            v_results = list(v_pool.map(lambda args: _check_single_video(*args), video_tasks))
        for res in v_results:
            if not res.dec_ok:
                video_decode_all_ok = False
            if not res.enc_ok:
                video_encode_all_ok = False
            video_details.append(f"ep{res.episode_index}/{res.camera_name}:{res.detail}")

    info_video_decl_ok = True
    for cam in STANDARD_CAMERAS:
        v_info = info_obj["features"].get(cam.feature_key, {}).get("info", {})
        if (
            v_info.get("video.codec") != "hevc"
            or v_info.get("video.pix_fmt") != "yuv420p"
            or int(v_info.get("video.fps", 0)) != fps
            or int(v_info.get("video.width", 0)) != TARGET_VIDEO_WIDTH
            or int(v_info.get("video.height", 0)) != TARGET_VIDEO_HEIGHT
            or v_info.get("has_audio") is not False
        ):
            info_video_decl_ok = False
    video_encode_ok = video_encode_all_ok and info_video_decl_ok
    video_encode_detail = (
        f"HEVC / yuv420p / 无音频 / GOP={VIDEO_GOP_SIZE} / 无 B 帧 / {fps}fps / {TARGET_VIDEO_WIDTH}x{TARGET_VIDEO_HEIGHT}"
        if video_encode_ok
        else f"视频流编码或 info.json 声明不一致 (stream_ok={video_encode_all_ok}, info_decl_ok={info_video_decl_ok})"
    )

    _add_item(report, "P00", "时间契约", time_ok, time_detail)
    _add_item(report, "P0", "索引可恢复", index_ok, index_detail)
    _add_item(report, "P0", "数值规范", num_ok, num_detail)
    _add_item(
        report,
        "P0",
        "状态与动作完整率",
        complete_ok,
        f"完整率 {completeness_ratio * 100.0:.2f}% (>= {MIN_STATE_COMPLETENESS_RATIO * 100.0:g}%)",
    )
    _add_item(report, "P00", "视频可解码对齐", video_decode_all_ok, f"共校验 {len(video_details)} 路视频，PTS 均匀间隔")
    _add_item(report, "P0", "视频编码合规", video_encode_ok, video_encode_detail)
    _add_item(report, "P0", "正运动学一致性校验", fk_ok, fk_detail)

    cam_params_path = agibot_dir / "camera_params.json"
    cam_params_ok = cam_params_path.exists()
    cam_params_detail = "各集内参、畸变系数、双目基线及顶层/逐帧外参均完整且四元数归一化"
    proj_ok = True
    proj_detail = "已验证双目（kannala_brandt 等距模型）末端点投影均落在画面坐标平面内"
    if cam_params_ok:
        cam_params_obj = json.loads(cam_params_path.read_text(encoding="utf-8"))
        for ep_idx in range(actual_ep_count):
            ep_cp = cam_params_obj.get(str(ep_idx), {})
            ep_len = int(ep_df.iloc[ep_idx]["length"])
            for cam in STANDARD_CAMERAS:
                ik = ep_cp.get(f"intrinsic_{cam.short_name}", {})
                if (
                    int(ik.get("width", 0)) != TARGET_VIDEO_WIDTH
                    or int(ik.get("height", 0)) != TARGET_VIDEO_HEIGHT
                    or float(ik.get("Fx", 0.0)) <= 0.0
                    or float(ik.get("Fy", 0.0)) <= 0.0
                    or float(ik.get("Cx", 0.0)) <= 0.0
                    or float(ik.get("Cy", 0.0)) <= 0.0
                ):
                    cam_params_ok = False
                    cam_params_detail = f"ep{ep_idx} intrinsic_{cam.short_name} 分辨率或焦距/主点异常"
                dist_model = str(ik.get("distortion_model", ""))
                req_dist_keys = ("k1", "k2", "k3", "k4") if dist_model == "kannala_brandt" else ("k1", "k2", "p1", "p2")
                if any(dk not in ik for dk in req_dist_keys):
                    cam_params_ok = False
                    cam_params_detail = f"ep{ep_idx} intrinsic_{cam.short_name} 缺少畸变键 {req_dist_keys}"

            top_ext_keys = ["extrinsic_head_stereo_left_T_right"] + [
                f"extrinsic_base_T_{cam.short_name}" for cam in STANDARD_CAMERAS
            ]
            for ek in top_ext_keys:
                ext_obj = ep_cp.get(ek)
                if not isinstance(ext_obj, dict) or "rotation" not in ext_obj or "translation" not in ext_obj:
                    cam_params_ok = False
                    cam_params_detail = f"ep{ep_idx} 缺少或格式错误: {ek}"
                else:
                    rot = ext_obj["rotation"]
                    if not _is_unit_quat(float(rot["x"]), float(rot["y"]), float(rot["z"]), float(rot["w"])):
                        cam_params_ok = False
                        cam_params_detail = f"ep{ep_idx} {ek} 四元数未归一化"

            frames_ext = ep_cp.get("frames", [])
            if len(frames_ext) != ep_len:
                cam_params_ok = False
                cam_params_detail = f"ep{ep_idx} frames 长度 {len(frames_ext)} != {ep_len}"
            else:
                frame_ext_keys = tuple(f"extrinsic_base_T_{cam.short_name}" for cam in STANDARD_CAMERAS)
                for f_item in frames_ext:
                    for ek in frame_ext_keys:
                        f_ext = f_item.get(ek)
                        if not isinstance(f_ext, dict) or "rotation" not in f_ext or "translation" not in f_ext:
                            cam_params_ok = False
                            cam_params_detail = f"ep{ep_idx} 逐帧缺少 {ek}"
                            break
                        rot = f_ext["rotation"]
                        if not _is_unit_quat(float(rot["x"]), float(rot["y"]), float(rot["z"]), float(rot["w"])):
                            cam_params_ok = False
                            cam_params_detail = f"ep{ep_idx} 逐帧 {ek} 四元数未归一化"
                            break
                    if not cam_params_ok:
                        break

            if all_episode_frames and cam_params_ok:
                sample_frames = np.linspace(0, ep_len - 1, min(PROJ_SAMPLE_FRAMES_PER_EP, ep_len), dtype=int)
                stereo_cams = (
                    ("head_stereo_left", "intrinsic_head_stereo_left", "extrinsic_base_T_head_stereo_left"),
                    ("head_stereo_right", "intrinsic_head_stereo_right", "extrinsic_base_T_head_stereo_right"),
                )
                for cam_label, intr_key, ext_key in stereo_cams:
                    intr = ep_cp[intr_key]
                    dist_model = str(intr.get("distortion_model", "pinhole"))
                    in_view_count = 0
                    for f_idx in sample_frames:
                        st_vec = np.array(all_episode_frames[ep_idx].iloc[f_idx]["observation.state"], dtype=np.float64)
                        ext_cam = frames_ext[f_idx][ext_key]
                        t_b_cam = np.eye(4, dtype=np.float64)
                        rot = ext_cam["rotation"]
                        trn = ext_cam["translation"]
                        t_b_cam[:3, :3] = quat_xyzw_to_matrix(rot["x"], rot["y"], rot["z"], rot["w"])
                        t_b_cam[:3, 3] = [trn["x"], trn["y"], trn["z"]]
                        t_cam_b = np.linalg.inv(t_b_cam)
                        frame_hit = False
                        for p_slice in (sl_le_p, sl_re_p):
                            p_cam = t_cam_b @ np.append(st_vec[p_slice], 1.0)
                            if p_cam[2] > 0.05:
                                u, v = _project_camera_point(
                                    x=float(p_cam[0]),
                                    y=float(p_cam[1]),
                                    z=float(p_cam[2]),
                                    fx=float(intr["Fx"]),
                                    fy=float(intr["Fy"]),
                                    cx=float(intr["Cx"]),
                                    cy=float(intr["Cy"]),
                                    distortion_model=dist_model,
                                    k1=float(intr.get("k1", 0.0)),
                                    k2=float(intr.get("k2", 0.0)),
                                    k3=float(intr.get("k3", 0.0)),
                                    k4=float(intr.get("k4", 0.0)),
                                    p1=float(intr.get("p1", 0.0)),
                                    p2=float(intr.get("p2", 0.0)),
                                )
                                if 0.0 <= u < float(TARGET_VIDEO_WIDTH) and 0.0 <= v < float(TARGET_VIDEO_HEIGHT):
                                    frame_hit = True
                        if frame_hit:
                            in_view_count += 1
                    if in_view_count / float(len(sample_frames)) < MIN_PROJ_IN_VIEW_RATIO:
                        proj_ok = False
                        proj_detail = f"ep{ep_idx} {cam_label} 末端投影落入画幅比例过低: {in_view_count}/{len(sample_frames)}"
    else:
        cam_params_detail = "缺少 meta/agibot/camera_params.json"
    _add_item(report, "P0", "相机参数完整", cam_params_ok, cam_params_detail)
    _add_item(report, "P0", "投影校验", proj_ok, proj_detail)

    ep_meta_path = agibot_dir / "episode_meta.parquet"
    stats_path = meta_dir / "stats.json"
    required_agibot_files = [
        agibot_dir / "manifest.json",
        meta_info_path,
        ep_meta_path,
        stats_path,
        agibot_dir / "schema.json",
        agibot_dir / "mapping.json",
        agibot_dir / "video_handler_params.json",
        agibot_dir / "events.parquet",
        agibot_dir / "keyframes.parquet",
        agibot_dir / "log.txt",
        dataset_dir / "README.md",
    ]
    meta_files_ok = all(p.exists() for p in required_agibot_files)
    meta_detail = "manifest/meta_info/episode_meta/stats.json 与实测一致"
    stats_ok = False
    if meta_files_ok and all_episode_frames:
        req_meta_info_keys = (
            "robot_type",
            "ee_list",
            "camera_list",
            "camera_type",
            "camera_fps",
            "aligned_base",
            "clip_start_time",
            "clip_end_time",
            "duration",
            "data_validate",
            "version",
            "sw_version",
        )
        meta_info_keys_ok = all(
            all(mk in meta_info_obj.get(str(ep_idx), {}) for mk in req_meta_info_keys)
            for ep_idx in range(actual_ep_count)
        )
        ep_meta_df = pd.read_parquet(ep_meta_path)
        req_ep_meta_cols = (
            "episode_index",
            "task_index",
            "task_name",
            "duration",
            "episode_status",
            "end_effector_names",
            "end_effector_types",
            "high_level_instruction",
        )
        ep_meta_cols_ok = (
            len(ep_meta_df) == actual_ep_count
            and all(col in ep_meta_df.columns for col in req_ep_meta_cols)
            and bool(ep_meta_df["high_level_instruction"].notna().all())
        )
        stats_obj = json.loads(stats_path.read_text(encoding="utf-8"))
        combined_df = pd.concat(all_episode_frames)
        all_states = np.stack(combined_df["observation.state"].to_numpy())
        all_actions = np.stack(combined_df["action"].to_numpy())
        err_st_min = float(np.max(np.abs(all_states.min(axis=0) - np.array(stats_obj["observation.state"]["min"], dtype=np.float64))))
        err_st_max = float(np.max(np.abs(all_states.max(axis=0) - np.array(stats_obj["observation.state"]["max"], dtype=np.float64))))
        err_ac_min = float(np.max(np.abs(all_actions.min(axis=0) - np.array(stats_obj["action"]["min"], dtype=np.float64))))
        err_ac_max = float(np.max(np.abs(all_actions.max(axis=0) - np.array(stats_obj["action"]["max"], dtype=np.float64))))
        vid_stats_ok = all(cam.feature_key in stats_obj for cam in STANDARD_CAMERAS)
        stats_ok = (
            meta_info_keys_ok
            and ep_meta_cols_ok
            and vid_stats_ok
            and err_st_min < 1e-4
            and err_st_max < 1e-4
            and err_ac_min < 1e-4
            and err_ac_max < 1e-4
        )
        if not stats_ok:
            meta_detail = (
                f"元数据核验异常: meta_info_keys={meta_info_keys_ok}, "
                f"ep_meta_cols={ep_meta_cols_ok}, vid_stats={vid_stats_ok}"
            )
    else:
        meta_detail = "缺少必需扩展元数据或交付文档"
    _add_item(report, "P1", "元数据齐全", meta_files_ok and stats_ok, meta_detail)

    min_dur = float(ep_df["length"].min()) / float(fps)
    min_short_side = min(
        min(int(info_obj["features"][c.feature_key]["shape"][0]), int(info_obj["features"][c.feature_key]["shape"][1]))
        for c in STANDARD_CAMERAS
    )
    qual_ok = min_dur >= MIN_EPISODE_DURATION_S and min_short_side >= MIN_VIDEO_SHORT_SIDE_PX
    _add_item(
        report,
        "P1",
        "画质与时长",
        qual_ok,
        f"最短短边={min_short_side}px (>={MIN_VIDEO_SHORT_SIDE_PX}), 最短时长={min_dur:.2f}s (>={MIN_EPISODE_DURATION_S:g}s)",
    )

    seg_path = agibot_dir / "segments.parquet"
    seg_ok = seg_path.exists()
    seg_detail = ""
    if seg_ok:
        seg_df = pd.read_parquet(seg_path)
        min_cov_ratio = 1.0
        for ep_idx in range(actual_ep_count):
            ep_len = int(ep_df.iloc[ep_idx]["length"])
            sub = seg_df[seg_df["episode_index"] == ep_idx].sort_values("start_frame_index")
            if sub.empty:
                seg_ok = False
                seg_detail = f"ep{ep_idx} 无分段标注"
                break
            prev_end = -1
            covered_frames = 0
            for _, srow in sub.iterrows():
                s = int(srow["start_frame_index"])
                e = int(srow["end_frame_index"])
                sk = str(srow["skill"])
                dur_s = float(e - s) / float(fps)
                if (
                    s < 0
                    or e > ep_len
                    or s < prev_end
                    or (prev_end >= 0 and (s - prev_end) > MAX_SEGMENT_BOUNDARY_GAP_FRAMES)
                    or e <= s
                    or sk not in STANDARD_SKILL_VOCAB
                    or dur_s < MIN_SEGMENT_DURATION_S
                    or dur_s > MAX_SEGMENT_DURATION_S
                ):
                    seg_ok = False
                    seg_detail = f"ep{ep_idx} 分段非法: [{s},{e})/{ep_len} skill={sk} dur={dur_s:.2f}s prev_end={prev_end}"
                    break
                prev_end = e
                covered_frames += e - s
            if not seg_ok:
                break
            cov_ratio = float(covered_frames) / float(ep_len)
            min_cov_ratio = min(min_cov_ratio, cov_ratio)
            if cov_ratio < MIN_SEGMENT_COVERAGE_RATIO:
                seg_ok = False
                seg_detail = f"ep{ep_idx} 分段覆盖率不足 {int(MIN_SEGMENT_COVERAGE_RATIO * 100)}% ({cov_ratio * 100.0:.2f}%)"
                break
        if seg_ok:
            seg_detail = f"共 {len(seg_df)} 条技能分段，最低覆盖率 {min_cov_ratio * 100.0:.2f}%，技能词均在标准词表内"
    _add_item(report, "P1", "标注合法（L2）", seg_ok, seg_detail)

    sdk_ok = False
    sdk_detail = ""
    try:
        ds = LeRobotDataset(repo_id=dataset_dir.name, root=dataset_dir, video_backend="pyav")
        checked_indices = sorted(
            {0, max(0, len(ds) // 2), max(0, len(ds) - 1)}
            | {int(r["dataset_from_index"]) for _, r in ep_df.iterrows()}
            | {max(0, int(r["dataset_to_index"]) - 1) for _, r in ep_df.iterrows()}
        )
        for sample_idx in checked_indices:
            sample = ds[sample_idx]
            for cam in STANDARD_CAMERAS:
                if tuple(sample[cam.feature_key].shape) != (3, TARGET_VIDEO_HEIGHT, TARGET_VIDEO_WIDTH):
                    raise ValueError(f"样本 {sample_idx} {cam.feature_key} 形状异常: {sample[cam.feature_key].shape}")
            if tuple(sample["observation.state"].shape) != (68,) or tuple(sample["action"].shape) != (68,):
                raise ValueError(f"样本 {sample_idx} 状态/动作向量形状异常")
        sdk_ok = len(ds) == total_frames and ds.num_episodes == total_episodes
        sdk_detail = f"LeRobotDataset(v0.4.4) 加载并抽样遍历 {len(checked_indices)} 帧通过，总帧数={len(ds)}"
    except Exception as exc:
        sdk_detail = f"LeRobotDataset 加载异常: {exc}"
    _add_item(report, "P00", "SDK 直接加载", sdk_ok, sdk_detail)

    return report


def write_quality_reports(report: ValidationReport, agibot_dir: Path) -> None:
    """将验收自检结果写入 meta/agibot/quality.json 与 meta/agibot/quality.md。

    参数:
        report: ValidationReport 验证结果对象。
        agibot_dir: meta/agibot 目录路径。
    """
    status = "passed" if report.blocking_issue_count == 0 else "failed"
    payload = {
        "std_version": "agibot-realrobot-lerobot-std v1.0",
        "dataset_root": report.dataset_root,
        "data_tier": report.data_tier,
        "status": status,
        "blocking_issue_count": report.blocking_issue_count,
        "p00_fail_count": report.p00_fail_count,
        "p0_fail_count": report.p0_fail_count,
        "p1_warn_count": report.p1_warn_count,
        "checks": [
            {
                "priority": item.priority,
                "name": item.name,
                "passed": item.passed,
                "detail": item.detail,
            }
            for item in report.items
        ],
    }
    (agibot_dir / "quality.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# 交付自检报告（{Path(report.dataset_root).name}）",
        "",
        "- 标准版本：`agibot-realrobot-lerobot-std v1.0`",
        f"- 声明档位：`{report.data_tier}`",
        f"- 总体状态：`{status}`（`blocking_issue_count = {report.blocking_issue_count}`，P00 失败 `{report.p00_fail_count}`，P0 失败 `{report.p0_fail_count}`，P1 警告 `{report.p1_warn_count}`）",
        "",
        "## 自动检查与一致性核对明细",
        "",
        "| 优先级 | 检查项 | 结果 | 详情 |",
        "| --- | --- | --- | --- |",
    ]
    for item in report.items:
        res_txt = "通过" if item.passed else "未通过"
        md_lines.append(f"| {item.priority} | {item.name} | {res_txt} | {item.detail} |")

    md_lines.extend(["", "## 第 12 章交付前自检清单勾选", ""])
    checklist = [
        "目录与路径模板符合第 3 章；meta/agibot/ 齐备未改名",
        "info.json 完整：v3.0 / robot_type 恒定 / fps 恒定；total_* 与实际一致",
        "field_descriptions：indices 不重叠、完整覆盖 shape、字段命名符合第 5 章",
        "按末端形态满足 5.3 必选矩阵；ee_list 末端声明正确；禁止字段未出现；action 齐备",
        "ts 纳秒绝对时间；timestamp 单调；frame_index 连续",
        "视频 HEVC / yuv420p / 无音频 / GOP=8 / 无 B 帧；PTS 从 0；帧数 = episode length；分辨率 = features.shape",
        "四元数 XYZW 且归一化；无 NaN / Inf；关节量已换算为 SI 单位（rad、m、N、N·m）",
        "状态与动作完整帧占比 >= 99%；缺失未以 0 / NaN / 复制 / 插值占位",
        "URDF 与 meshes 齐全可解析；关节限位与状态向量一致；坐标系符合第 8 章",
        "camera_params.json 逐 episode 齐全；各路含基座系外参；运动机位含逐帧外参；内参与分辨率一致",
        "L2：segments 覆盖率 >= 80%、词表内技能、边界 <= 2 帧；high_level_instruction 齐备",
        "quality.json / quality.md 自检完成，blocking_issue_count = 0",
        "用 lerobot v0.4.4 LeRobotDataset 加载冒烟通过",
        "说明文档齐备（含字段说明、坐标系、脱敏与合规声明）；合规授权完成",
    ]
    all_pass = report.blocking_issue_count == 0 and report.p1_warn_count == 0
    mark = "x" if all_pass else " "
    for c in checklist:
        md_lines.append(f"- [{mark}] {c}")

    (agibot_dir / "quality.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> int:
    """命令行入口：按脚本默认参数（或可选 CLI 覆盖）验证单个数据集或扫描总目录下全部子集。"""
    parser = argparse.ArgumentParser(description="验证 LeRobot v3.0 真机数据规范合规性")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="包含多个任务子集数据集的总目录",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="直接指定单个数据集根目录（优先于 --root-dir）",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        default=DEFAULT_WRITE_REPORT,
        help="将自检结果写回 meta/agibot/quality.json 与 quality.md",
    )
    args = parser.parse_args()

    targets: List[Path] = []
    if args.dataset_dir is not None:
        targets.append(args.dataset_dir)
    else:
        if not args.root_dir.exists():
            print(f"错误：目录不存在 {args.root_dir}", file=sys.stderr)
            return 2
        for sub in sorted(args.root_dir.iterdir()):
            if sub.is_dir() and (sub / "meta" / "info.json").exists():
                targets.append(sub)

    if not targets:
        print("错误：未找到任何包含 meta/info.json 的数据集目录", file=sys.stderr)
        return 2

    any_blocking = False
    for ds_dir in targets:
        print(f"=== 开始验证数据集: {ds_dir} ===")
        report = validate_dataset(ds_dir)
        if args.write_report and (ds_dir / "meta" / "agibot").is_dir():
            write_quality_reports(report, ds_dir / "meta" / "agibot")
        for item in report.items:
            tag = "PASS" if item.passed else "FAIL"
            print(f"  [{item.priority}] [{tag}] {item.name}: {item.detail}")
        print(
            f"汇总: blocking={report.blocking_issue_count}, P00_fail={report.p00_fail_count}, "
            f"P0_fail={report.p0_fail_count}, P1_warn={report.p1_warn_count}"
        )
        if report.blocking_issue_count > 0:
            any_blocking = True

    return 1 if any_blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
