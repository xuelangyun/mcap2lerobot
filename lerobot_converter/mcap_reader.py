# -*- coding: utf-8 -*-
"""SYNC.mcap 文件单遍解析（URDF、内外参、时间戳、关节与里程计）及硬件/并发 HEVC 编码模块。"""

from __future__ import annotations

import functools
import math
import struct
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import zstandard
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from .config import (
    MCAP_PIPE_X265_POOLS,
    VIDEO_CRF_OR_QP,
    VIDEO_GOP_SIZE,
    VIDEO_MAX_B_FRAMES,
    X265_POOLS_PER_STREAM,
)
from .kinematics import matrix_to_transform_pose
from .lerobot_spec import (
    DEFAULT_VIDEO_CODEC,
    DEFAULT_X265_PRESET,
    STANDARD_CAMERAS,
    TARGET_VIDEO_FPS,
    TARGET_VIDEO_HEIGHT,
    TARGET_VIDEO_WIDTH,
    CameraIntrinsics,
    EpisodeMcapParsed,
    McapJointSeries,
    McapOdomSeries,
    TransformPose,
)


@functools.lru_cache(maxsize=8)
def detect_hevc_encoder(preferred_vcodec: str = DEFAULT_VIDEO_CODEC) -> str:
    """探测并返回当前环境实际可用的高速 HEVC 编码器。

    参数:
        preferred_vcodec: 指定编码器名称或 'auto' 自动探测。
    返回值:
        选定的 ffmpeg 编码器名（如 'hevc_amf'、'hevc_nvenc'、'hevc_qsv' 或 'libx265'）。
    """
    if preferred_vcodec != "auto":
        return preferred_vcodec

    # 优先探测硬件编码器（通过实际单帧试编码验证，防止仅列举但运行失败）
    for candidate in ("hevc_nvenc", "hevc_amf", "hevc_qsv"):
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={TARGET_VIDEO_WIDTH}x{TARGET_VIDEO_HEIGHT}:d=0.04:r={TARGET_VIDEO_FPS}",
            "-frames:v",
            "1",
            "-c:v",
            candidate,
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(VIDEO_GOP_SIZE),
            "-bf",
            str(VIDEO_MAX_B_FRAMES),
            "-f",
            "null",
            "-",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=8)
            if res.returncode == 0:
                return candidate
        except Exception:
            continue
    return "libx265"


def build_hevc_encoder_args(
    vcodec: str,
    x265_preset: str = DEFAULT_X265_PRESET,
    vf_expr: Optional[str] = None,
    pools: int = X265_POOLS_PER_STREAM,
    is_mjpeg_pipe: bool = False,
) -> List[str]:
    """构建严格满足规范第 6 章（HEVC/yuv420p/GOP/无B帧/无音频）的 ffmpeg 输出参数。

    参数:
        vcodec: 已选定的 HEVC 编码器名。
        x265_preset: libx265 预设档位。
        vf_expr: 可选视频滤镜表达式。
        pools: libx265 线程池大小（并发编码时避免过度抢占核心）。
        is_mjpeg_pipe: 输入是否为 MJPEG 管道流（需显式指定 tv 色域防止 yuvj420p）。
    返回值:
        ffmpeg 命令行输出参数列表。
    """
    args: List[str] = []
    if vf_expr:
        args.extend(["-vf", vf_expr])
    args.extend(["-c:v", vcodec, "-pix_fmt", "yuv420p"])
    if is_mjpeg_pipe:
        args.extend(["-color_range", "tv"])

    qp_str = str(VIDEO_CRF_OR_QP)
    gop_str = str(VIDEO_GOP_SIZE)
    bf_str = str(VIDEO_MAX_B_FRAMES)
    if vcodec == "libx265":
        args.extend(
            [
                "-preset",
                x265_preset,
                "-crf",
                qp_str,
                "-x265-params",
                f"bframes={bf_str}:scenecut=0:keyint={gop_str}:min-keyint={gop_str}:open-gop=0:pools={pools}:log-level=error",
            ]
        )
    elif vcodec == "hevc_amf":
        args.extend(["-rc", "cqp", "-qp_i", qp_str, "-qp_p", qp_str])
    elif vcodec == "hevc_nvenc":
        args.extend(["-preset", "p4", "-rc", "constqp", "-qp", qp_str])
    elif vcodec == "hevc_qsv":
        args.extend(["-global_quality", qp_str])

    args.extend(
        [
            "-g",
            gop_str,
            "-keyint_min",
            gop_str,
            "-bf",
            bf_str,
            "-sc_threshold",
            "0",
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            "-1",
            "-an",
            "-tag:v",
            "hvc1",
        ]
    )
    return args


def _read_len_bytes(buf: bytes, offset: int) -> Tuple[bytes, int]:
    """从缓冲区读取 4 字节小端长度前缀的字节切片。"""
    ln = struct.unpack_from("<I", buf, offset)[0]
    start = offset + 4
    return buf[start : start + ln], start + ln


def _read_len_bytes64(buf: bytes, offset: int) -> Tuple[bytes, int]:
    """从缓冲区读取 8 字节小端长度前缀的字节切片。"""
    ln = struct.unpack_from("<Q", buf, offset)[0]
    start = offset + 8
    return buf[start : start + ln], start + ln


def _read_len_str(buf: bytes, offset: int) -> Tuple[str, int]:
    """从缓冲区读取 4 字节小端长度前缀的 UTF-8 字符串。"""
    raw, next_off = _read_len_bytes(buf, offset)
    return raw.decode("utf-8", errors="replace"), next_off


def _decode_varint(buf: bytes, offset: int) -> Tuple[int, int]:
    """解码 Protobuf Base-128 Varint 整数并返回新偏移量。"""
    shift = 0
    val = 0
    pos = offset
    while True:
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7


def _extract_compressed_image_ts_fast(payload: bytes, header_cls, img_cls) -> int:
    """快速从 CompressedImage Protobuf 字节中仅解析 header 时间戳，跳过图像字节拷贝。"""
    if payload and payload[0] == 0x0A:
        hdr_len, hdr_start = _decode_varint(payload, 1)
        hdr = header_cls()
        hdr.ParseFromString(payload[hdr_start : hdr_start + hdr_len])
        return int(hdr.timestamp.sec) * 1_000_000_000 + int(hdr.timestamp.nanosec)
    msg = img_cls()
    msg.ParseFromString(payload)
    return int(msg.header.timestamp.sec) * 1_000_000_000 + int(msg.header.timestamp.nanosec)


def _build_camera_intrinsics(
    camera_name: str,
    width: int,
    height: int,
    distortion_model: str,
    camera_type: str,
    k_flat: List[float],
    d_list: List[float],
) -> CameraIntrinsics:
    """根据 MCAP CameraInfo 构造内参对象并缩放至统一交付分辨率。"""
    sx = float(TARGET_VIDEO_WIDTH) / float(width)
    sy = float(TARGET_VIDEO_HEIGHT) / float(height)
    return CameraIntrinsics(
        camera_name=camera_name,
        src_width=int(width),
        src_height=int(height),
        width=TARGET_VIDEO_WIDTH,
        height=TARGET_VIDEO_HEIGHT,
        fx=float(k_flat[0]) * sx,
        fy=float(k_flat[4]) * sy,
        cx=float(k_flat[2]) * sx,
        cy=float(k_flat[5]) * sy,
        distortion_model=distortion_model,
        camera_type=camera_type,
        distortion_coeffs=[float(x) for x in d_list],
    )


def _open_mcap_metadata(mcap_path: Path) -> Tuple[int, dict, dict, descriptor_pool.DescriptorPool]:
    """读取 MCAP 尾部索引，构建通道映射与 Protobuf 消息类映射。

    参数:
        mcap_path: SYNC.mcap 文件路径。
    返回值:
        (summary_start 偏移, channels 映射, msg_classes 映射, Protobuf 描述池)。
    """
    with open(mcap_path, "rb") as f:
        f.seek(0, 2)
        fsize = f.tell()
        f.seek(fsize - 37)
        f.read(9)
        summary_start, summary_offset_start, _ = struct.unpack("<QQI", f.read(20))
        f.seek(summary_start)
        end_pos = summary_offset_start if summary_offset_start > 0 else fsize - 37
        schemas = {}
        channels = {}
        while f.tell() < end_pos:
            op = f.read(1)[0]
            rlen = struct.unpack("<Q", f.read(8))[0]
            data = f.read(rlen)
            if op == 0x03:
                sid = struct.unpack_from("<H", data, 0)[0]
                sname, off = _read_len_str(data, 2)
                enc, off = _read_len_str(data, off)
                sdata, off = _read_len_bytes(data, off)
                schemas[sid] = (sname, enc, sdata)
            elif op == 0x04:
                cid, sid = struct.unpack_from("<HH", data, 0)
                topic, off = _read_len_str(data, 4)
                channels[cid] = (sid, topic)

    pool = descriptor_pool.DescriptorPool()
    all_files = {}
    for _, (_, _, sdata) in schemas.items():
        fds = descriptor_pb2.FileDescriptorSet()
        fds.ParseFromString(sdata)
        for fd in fds.file:
            all_files[fd.name] = fd
    added = set()

    def add_proto_file(fname: str) -> None:
        if fname in added:
            return
        for dep in all_files[fname].dependency:
            add_proto_file(dep)
        pool.Add(all_files[fname])
        added.add(fname)

    for fname in all_files:
        add_proto_file(fname)

    msg_classes = {
        sid: message_factory.GetMessageClass(pool.FindMessageTypeByName(sname))
        for sid, (sname, _, _) in schemas.items()
    }
    return summary_start, channels, msg_classes, pool


def _extract_group_pv(sensor_msg, group_name: str, dim: int) -> Tuple[List[float], List[float]]:
    """从 SingoriXSensor 消息中提取指定关节组的位置与速度列表。"""
    entry = sensor_msg.joint_sensor_map[group_name]
    pos = [float(x) for x in entry.position[:dim]]
    vel = [float(x) for x in entry.velocity[:dim]]
    if len(pos) < dim:
        pos.extend([0.0] * (dim - len(pos)))
    if len(vel) < dim:
        vel.extend([0.0] * (dim - len(vel)))
    return pos, vel


def _spawn_mjpeg_ffmpeg_pipe(
    dst_mp4: Path,
    src_width: int,
    src_height: int,
    resolved_vcodec: str,
    x265_preset: str,
) -> subprocess.Popen:
    """启动单路接收 MJPEG 管道输入的 ffmpeg HEVC 编码进程。"""
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if src_width != TARGET_VIDEO_WIDTH or src_height != TARGET_VIDEO_HEIGHT:
        vf_expr = f"scale={TARGET_VIDEO_WIDTH}:{TARGET_VIDEO_HEIGHT}:flags=lanczos,format=yuv420p"
    else:
        vf_expr = "format=yuv420p"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-framerate",
        str(TARGET_VIDEO_FPS),
        "-i",
        "-",
    ]
    cmd.extend(
        build_hevc_encoder_args(
            vcodec=resolved_vcodec,
            x265_preset=x265_preset,
            vf_expr=vf_expr,
            pools=MCAP_PIPE_X265_POOLS,
            is_mjpeg_pipe=True,
        )
    )
    cmd.append(str(dst_mp4))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def parse_sync_mcap(
    mcap_path: Path,
    include_state_streams: bool = True,
    dst_video_paths: Optional[List[Path]] = None,
    vcodec: str = DEFAULT_VIDEO_CODEC,
    x265_preset: str = DEFAULT_X265_PRESET,
) -> EpisodeMcapParsed:
    """单遍解析 SYNC.mcap 中的标定、URDF、时间戳、可选状态流及可选四路视频直出编码。

    参数:
        mcap_path: SYNC.mcap 文件绝对路径。
        include_state_streams: 是否同步从 MCAP 提取关节（singorix/wbcs/sensor）与里程计（/odom/base_link）。
        dst_video_paths: 非空时，在同一次遍历内把四路 JPEG 流经管道直接编码到目标 MP4，避免二次解压 MCAP。
        vcodec: HEVC 编码器名称或 'auto'。
        x265_preset: libx265 预设档位。
    返回值:
        EpisodeMcapParsed 强类型解析结果。
    """
    summary_start, channels, msg_classes, pool = _open_mcap_metadata(mcap_path)
    header_cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("galbot.core_proto.Header"))
    resolved_vcodec = detect_hevc_encoder(vcodec) if dst_video_paths is not None else ""

    urdf_xml = ""
    robot_sn = ""
    hw_ver = ""
    sw_ver = ""
    hl_timestamps: List[int] = []
    cam_info_map = {}
    tf_static_map = {}
    stereo_tf_fallback: Optional[TransformPose] = None
    stereo_t_mat_raw: Optional[List[float]] = None

    j_ts_list: List[int] = []
    la_p_list: List[List[float]] = []
    la_v_list: List[List[float]] = []
    ra_p_list: List[List[float]] = []
    ra_v_list: List[List[float]] = []
    lg_p_list: List[float] = []
    rg_p_list: List[float] = []
    hd_p_list: List[List[float]] = []
    hd_v_list: List[List[float]] = []
    leg_p_list: List[List[float]] = []
    leg_v_list: List[List[float]] = []

    odom_ts_list: List[int] = []
    odom_pos_list: List[List[float]] = []
    odom_ori_list: List[List[float]] = []

    info_topic_to_short = {c.info_topic: c.short_name for c in STANDARD_CAMERAS}
    topic_to_cam_idx = {c.image_topic: idx for idx, c in enumerate(STANDARD_CAMERAS)}

    # 当启用单遍直出视频编码时，先暂存 camera_info 就绪前的首批帧，就绪后立即打开 4 路并发编码管道
    ffmpeg_procs: Optional[List[subprocess.Popen]] = None
    early_jpeg_bufs: List[List[bytes]] = [[] for _ in STANDARD_CAMERAS]
    written_counts: List[int] = [0] * len(STANDARD_CAMERAS)

    dctx = zstandard.ZstdDecompressor()
    try:
        with open(mcap_path, "rb") as f:
            f.seek(8)
            while f.tell() < summary_start:
                op = f.read(1)[0]
                rlen = struct.unpack("<Q", f.read(8))[0]
                if op != 0x06:
                    f.seek(rlen, 1)
                    continue
                chunk_hdr = f.read(rlen)
                off = 28
                comp, off = _read_len_str(chunk_hdr, off)
                records_bytes, off = _read_len_bytes64(chunk_hdr, off)
                raw = dctx.decompress(records_bytes) if comp == "zstd" else records_bytes

                pos = 0
                raw_len = len(raw)
                while pos < raw_len:
                    rop = raw[pos]
                    rec_len = struct.unpack_from("<Q", raw, pos + 1)[0]
                    if rop == 0x05:
                        cid = struct.unpack_from("<H", raw, pos + 9)[0]
                        sid, topic = channels[cid]
                        payload = raw[pos + 31 : pos + 9 + rec_len]

                        cam_idx = topic_to_cam_idx.get(topic)
                        if cam_idx is not None:
                            if dst_video_paths is not None:
                                msg = msg_classes[sid]()
                                msg.ParseFromString(payload)
                                if cam_idx == 0:
                                    ts_ns = int(msg.header.timestamp.sec) * 1_000_000_000 + int(
                                        msg.header.timestamp.nanosec
                                    )
                                    hl_timestamps.append(ts_ns)
                                if ffmpeg_procs is not None:
                                    ffmpeg_procs[cam_idx].stdin.write(msg.data)
                                else:
                                    early_jpeg_bufs[cam_idx].append(bytes(msg.data))
                                written_counts[cam_idx] += 1
                            elif cam_idx == 0:
                                hl_timestamps.append(
                                    _extract_compressed_image_ts_fast(payload, header_cls, msg_classes[sid])
                                )
                        elif include_state_streams and topic == "singorix/wbcs/sensor":
                            msg = msg_classes[sid]()
                            msg.ParseFromString(payload)
                            if "right_arm" in msg.joint_sensor_map and "left_arm" in msg.joint_sensor_map:
                                ts_ns = msg.header.timestamp.sec * 1_000_000_000 + msg.header.timestamp.nanosec
                                la_p, la_v = _extract_group_pv(msg, "left_arm", 7)
                                ra_p, ra_v = _extract_group_pv(msg, "right_arm", 7)
                                lg_p, _ = _extract_group_pv(msg, "left_gripper", 1)
                                rg_p, _ = _extract_group_pv(msg, "right_gripper", 1)
                                hd_p, hd_v = _extract_group_pv(msg, "head", 2)
                                leg_p, leg_v = _extract_group_pv(msg, "leg", 5)
                                j_ts_list.append(int(ts_ns))
                                la_p_list.append(la_p)
                                la_v_list.append(la_v)
                                ra_p_list.append(ra_p)
                                ra_v_list.append(ra_v)
                                lg_p_list.append(lg_p[0])
                                rg_p_list.append(rg_p[0])
                                hd_p_list.append(hd_p)
                                hd_v_list.append(hd_v)
                                leg_p_list.append(leg_p)
                                leg_v_list.append(leg_v)
                        elif include_state_streams and topic == "/odom/base_link":
                            msg = msg_classes[sid]()
                            msg.ParseFromString(payload)
                            ts_ns = msg.header.timestamp.sec * 1_000_000_000 + msg.header.timestamp.nanosec
                            p = msg.pose.pose.position
                            o = msg.pose.pose.orientation
                            qx, qy, qz, qw = float(o.x), float(o.y), float(o.z), float(o.w)
                            norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
                            ori = [0.0, 0.0, 0.0, 1.0] if norm < 1e-12 else [qx / norm, qy / norm, qz / norm, qw / norm]
                            odom_ts_list.append(int(ts_ns))
                            odom_pos_list.append([float(p.x), float(p.y)])
                            odom_ori_list.append(ori)
                        elif topic == "robot_description" and not urdf_xml:
                            msg = msg_classes[sid]()
                            msg.ParseFromString(payload)
                            urdf_xml = msg.value
                        elif topic == "/galbot/mission_manager/onboard_config" and not robot_sn:
                            msg = msg_classes[sid]()
                            msg.ParseFromString(payload)
                            robot_sn = msg.robot_serial_number
                            hw_ver = msg.robot_hardware_version
                            sw_ver = msg.mainline_version
                        elif topic in info_topic_to_short and topic not in cam_info_map:
                            msg = msg_classes[sid]()
                            msg.ParseFromString(payload)
                            cam_info_map[topic] = _build_camera_intrinsics(
                                camera_name=info_topic_to_short[topic],
                                width=msg.width,
                                height=msg.height,
                                distortion_model=msg.distortion_model,
                                camera_type=msg.camera_type,
                                k_flat=list(msg.k),
                                d_list=list(msg.d),
                            )
                            if topic == "/front_head_camera/left_color/camera_info" and len(msg.T) == 12:
                                t_vals = [float(x) for x in msg.T]
                                if any(abs(v) > 1e-9 for v in t_vals):
                                    stereo_t_mat_raw = t_vals
                            if dst_video_paths is not None and ffmpeg_procs is None and len(cam_info_map) == len(STANDARD_CAMERAS):
                                ordered_intr = [cam_info_map[c.info_topic] for c in STANDARD_CAMERAS]
                                ffmpeg_procs = [
                                    _spawn_mjpeg_ffmpeg_pipe(
                                        dst_mp4=dst_p,
                                        src_width=intr.src_width,
                                        src_height=intr.src_height,
                                        resolved_vcodec=resolved_vcodec,
                                        x265_preset=x265_preset,
                                    )
                                    for dst_p, intr in zip(dst_video_paths, ordered_intr, strict=True)
                                ]
                                for c_i, buf_list in enumerate(early_jpeg_bufs):
                                    for jpeg_bytes in buf_list:
                                        ffmpeg_procs[c_i].stdin.write(jpeg_bytes)
                                    buf_list.clear()
                        elif topic == "/embosa_tf_static":
                            msg = msg_classes[sid]()
                            msg.ParseFromString(payload)
                            for tr in msg.transforms:
                                t = tr.transform.translation
                                r = tr.transform.rotation
                                tf_static_map[(tr.header.frame_id, tr.child_frame_id)] = TransformPose(
                                    tx=float(t.x),
                                    ty=float(t.y),
                                    tz=float(t.z),
                                    qw=float(r.w),
                                    qx=float(r.x),
                                    qy=float(r.y),
                                    qz=float(r.z),
                                )
                        elif topic == "/embosa_tf" and stereo_tf_fallback is None:
                            msg = msg_classes[sid]()
                            msg.ParseFromString(payload)
                            for tr in msg.transforms:
                                if (
                                    tr.header.frame_id == "head_left_camera_color_optical_frame"
                                    and tr.child_frame_id == "head_right_camera_color_optical_frame"
                                ):
                                    t = tr.transform.translation
                                    r = tr.transform.rotation
                                    stereo_tf_fallback = TransformPose(
                                        tx=float(t.x),
                                        ty=float(t.y),
                                        tz=float(t.z),
                                        qw=float(r.w),
                                        qx=float(r.x),
                                        qy=float(r.y),
                                        qz=float(r.z),
                                    )
                    pos += 9 + rec_len
    finally:
        if ffmpeg_procs is not None:
            for idx, proc in enumerate(ffmpeg_procs):
                if proc.stdin is not None:
                    proc.stdin.close()
                _, err_bytes = proc.communicate()
                if proc.returncode != 0:
                    err_txt = err_bytes.decode("utf-8", errors="replace") if err_bytes else ""
                    raise RuntimeError(f"MCAP 相机 {STANDARD_CAMERAS[idx].short_name} 编码失败: {err_txt[-800:]}")

    if dst_video_paths is not None and any(cnt != len(hl_timestamps) for cnt in written_counts):
        raise RuntimeError(f"MCAP 各路相机写入帧数与基准帧数不一致: {written_counts} vs {len(hl_timestamps)}")

    if stereo_t_mat_raw is not None:
        mat_r_l = np.eye(4, dtype=np.float64)
        mat_r_l[:3, :4] = np.array(stereo_t_mat_raw, dtype=np.float64).reshape(3, 4)
        stereo_left_T_right = matrix_to_transform_pose(np.linalg.inv(mat_r_l))
    elif stereo_tf_fallback is not None:
        stereo_left_T_right = stereo_tf_fallback
    else:
        raise ValueError(f"未能从 {mcap_path} 中找到头部双目基线外参")

    joint_series: Optional[McapJointSeries] = None
    if include_state_streams and j_ts_list:
        j_ts_arr = np.array(j_ts_list, dtype=np.int64)
        order = np.argsort(j_ts_arr, kind="stable")
        joint_series = McapJointSeries(
            timestamps_ns=j_ts_arr[order],
            left_arm_pos=np.array(la_p_list, dtype=np.float64)[order],
            left_arm_vel=np.array(la_v_list, dtype=np.float64)[order],
            right_arm_pos=np.array(ra_p_list, dtype=np.float64)[order],
            right_arm_vel=np.array(ra_v_list, dtype=np.float64)[order],
            left_gripper_pct=np.array(lg_p_list, dtype=np.float64)[order],
            right_gripper_pct=np.array(rg_p_list, dtype=np.float64)[order],
            head_pos=np.array(hd_p_list, dtype=np.float64)[order],
            head_vel=np.array(hd_v_list, dtype=np.float64)[order],
            leg_pos=np.array(leg_p_list, dtype=np.float64)[order],
            leg_vel=np.array(leg_v_list, dtype=np.float64)[order],
        )

    odom_series: Optional[McapOdomSeries] = None
    if include_state_streams and odom_ts_list:
        o_ts_arr = np.array(odom_ts_list, dtype=np.int64)
        order = np.argsort(o_ts_arr, kind="stable")
        odom_series = McapOdomSeries(
            timestamps_ns=o_ts_arr[order],
            pos_xy=np.array(odom_pos_list, dtype=np.float64)[order],
            ori_xyzw=np.array(odom_ori_list, dtype=np.float64)[order],
        )

    ordered_intrinsics = [cam_info_map[c.info_topic] for c in STANDARD_CAMERAS]
    return EpisodeMcapParsed(
        urdf_xml=urdf_xml,
        robot_serial_number=robot_sn,
        hardware_version=hw_ver,
        sw_version=sw_ver,
        head_left_timestamps_ns=np.array(hl_timestamps, dtype=np.int64),
        camera_intrinsics=ordered_intrinsics,
        stereo_left_T_right=stereo_left_T_right,
        head_mount_T_head_left=tf_static_map[
            ("head_end_effector_mount_link", "head_left_camera_color_optical_frame")
        ],
        left_mount_T_hand_left=tf_static_map[
            ("left_arm_end_effector_mount_link", "left_arm_camera_color_optical_frame")
        ],
        right_mount_T_hand_right=tf_static_map[
            ("right_arm_end_effector_mount_link", "right_arm_camera_color_optical_frame")
        ],
        joint_series=joint_series,
        odom_series=odom_series,
    )
