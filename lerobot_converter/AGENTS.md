# AGENTS.md

本文件面向在 `lerobot_converter` 目录内进行开发、调试或自动化验证的编码智能体与维护人员。

## 模块职责

- `config.py`：全局参数变量集中维护模块，统一管理输入输出路径（`INPUT_ROOT`、`OUTPUT_ROOT`、`URDF_DESC_ROOT`）、`TASK_MAPPINGS`（`TaskMappingConfig`）、数据源与编码器开关（`VIDEO_SOURCE`、`STATE_SOURCE`、`VIDEO_CODEC`、`X265_PRESET`）、编码质量/GOP、并发线程池数及全部验收阈值。
- `lerobot_spec.py`：复用 `config.py` 规格常量，定义相机映射表 `STANDARD_CAMERAS`、强类型 `@dataclass` 数据结构（含 `MeshFileDigest`、`VideoCheckResult`）、68 维向量 `field_descriptions` 构造函数、技能词表归一化及流式分块哈希工具。
- `kinematics.py`：URDF XML 解析（`parse_urdf_joints`）、纯 NumPy 正运动学解算器 `UrdfKinematicsSolver`、位姿矩阵与四元数转换、URDF 网格多线程单遍拷贝+SHA256 计算与相对路径改写（`ensure_local_urdf_with_meshes`）。
- `mcap_reader.py`：`.SYNC.mcap` 单遍解析器 `parse_sync_mcap`（提取 URDF、相机内参、TF 外参、参考帧时间戳以及 `singorix/wbcs/sensor`、`/odom/base_link`，并在 `dst_video_paths` 非空时同步 4 路 `ffmpeg image2pipe` MJPEG 直出编码），以及 HEVC 硬件编码器自动探测 `detect_hevc_encoder` 与参数构造 `build_hevc_encoder_args`。
- `convert_to_lerobot.py`：批量转换入口，读取 `config.py` 中的路径/任务清单/开关，支持并发预解析 MCAP、4 路相机并发编码、PyAV `skip_frame="NONKEY"` 关键帧像素统计、Parquet 与紧凑 `camera_params.json` 写入、全局统计与 `episodes` 分集索引、`meta/agibot/` 扩展平面写入，并在末尾自动执行验收并回写 `quality.json`/`quality.md`。
- `validate_lerobot.py`：独立验收器，默认参数与阈值全部引自 `config.py`，使用合并单遍 PyAV `_check_single_video` 多线程并发检验视频编码/PTS/GOP，并多线程并发校验 URDF 网格 SHA256，覆盖 P00 / P0 / P1 全量验收项。
- `requirements.txt` / `environment.yml`：锁定 `Python 3.10.19`、`lerobot==0.4.4`、`yourdfpy==0.0.60`、`av==15.1.0`、`numpy==2.2.6`、`pandas==2.3.3`、`pyarrow==22.0.0`、`datasets==4.1.1`、`zstandard==0.25.0`、`protobuf==6.33.2` 等经过实测验证的精确版本，用于可复现环境初始化。

## 编码规范（强制）

1. **禁止使用松散字典作为函数参数**：函数与方法参数必须使用具名参数或 `@dataclass` 实体类传递；仅允许在最终 JSON / Parquet / 第三方库接口边界局部构造原生字典。
2. **中文注释**：所有类、dataclass 字段、函数及复杂逻辑必须使用中文 docstring 与中文行内注释说明用途、参数与返回值，禁止使用英文注释。
3. **局部编辑**：修改代码时仅格式化所改动的代码块，避免全文件无关重写。

## 关键实现约束

- **视频编解码约束**：所有输出视频必须满足 `codec=hevc`、`pix_fmt=yuv420p`、`has_b_frames=0`、`GOP=8`、无音频流、`fps=30`、分辨率 `640x480`。默认 `--vcodec auto` 通过 `detect_hevc_encoder` 自动探测硬件编码器并回退 `libx265`（预设由 `--x265-preset` 控制）。使用 MCAP `image2pipe` 解码 MJPEG 时，必须显式包含 `-vf format=yuv420p` 与 `-color_range tv`，否则 `ffmpeg` 会输出 `yuvj420p` 导致验收失败。
- **时间同步与 `LeRobotDataset` 容差**：`LeRobotDataset(tolerance_s=1e-4)` 要求视频 PTS 与 Parquet `timestamp` 在 0.1 ms 内对齐，因此帧时间网格严格按 `k / 30` 构造且 `ts` 与 `timestamp` 保持一致换算；关节与里程计测量值则按实际采集帧时间戳 `head_left_timestamps_ns` 做最近邻对齐。
- **URDF 网格**：交付目录下的 `meta/agibot/urdf/galbot_one_golf.urdf` 将 `package://galbot_one_golf_description/meshes/...` 改写为相对路径 `meshes/...`，确保 `yourdfpy.URDF.load` 在离线无 ROS 环境下直接加载成功。`ensure_local_urdf_with_meshes` 在多线程拷贝网格时同步完成 SHA256 计算并返回 `MeshFileDigest` 列表，避免二次读盘。

## 常用验证命令

在项目根目录 `f:\data-sample` 下使用 `lerobot` 环境运行：

```powershell
# 1. 默认模式转换（视频读 MP4，非视频状态优先读 MCAP，自动探测编码器）
D:\software\miniconda3\envs\lerobot\python.exe -m lerobot_converter.convert_to_lerobot

# 2. 纯 MCAP 模式验证（视频与状态均来自 SYNC.mcap）
D:\software\miniconda3\envs\lerobot\python.exe -m lerobot_converter.convert_to_lerobot --video-source mcap --state-source mcap

# 3. 独立验收全量数据集
D:\software\miniconda3\envs\lerobot\python.exe -m lerobot_converter.validate_lerobot --root-dir f:\data-sample\lerobot_output
```
