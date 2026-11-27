# LeRobot v3.0 真机数据转换与验收工具

将原始 Galbot G1 真机采集数据（`.SYNC.mcap`、`camera_*.mp4`、`joints.json`、`odom_base_link.json`、`task_info.json`）转换为符合《真机数据 LeRobot 格式标准与验收规范（外部供应商版）》L3 档位要求的 LeRobot v3.0 + `meta/agibot/` 数据集，并提供全自动验收脚本。

## 目录结构

```text
lerobot_converter/
├── __init__.py            # 工具包入口标记
├── config.py              # 全局参数变量集中维护模块（路径、任务清单、开关、编码、并发与验收阈值）
├── lerobot_spec.py        # 数据契约常量、强类型数据结构与特征描述构造
├── kinematics.py          # URDF 解析、正运动学（FK）解算与网格文件打包
├── mcap_reader.py         # SYNC.mcap 单遍解析（标定/URDF/时间戳/关节/里程计）及图像流直出编码
├── convert_to_lerobot.py  # 转换主程序（生成 Core Plane、meta/agibot/ 与自检报告）
├── validate_lerobot.py    # 独立验收脚本（P00 / P0 / P1 / FK / 投影 / LeRobotDataset 加载）
├── requirements.txt       # 精确锁定的 Python 依赖清单（pip）
├── environment.yml        # Conda 环境初始化定义文件
├── README.md              # 使用与规范说明
└── AGENTS.md              # 智能体开发与维护指南
```

## 环境初始化与明确版本

已在当前 Windows 环境完整验证以下版本组合（完整锁定清单见 [requirements.txt](requirements.txt) 与 [environment.yml](environment.yml)）：

| 组件 | 明确版本 | 作用说明 |
| --- | --- | --- |
| Python | `3.10.19` | 运行时解释器 |
| `lerobot` | `0.4.4` | LeRobot v3.0 数据写入、统计与 `LeRobotDataset` 校验 |
| `yourdfpy` / `trimesh` / `lxml` / `scipy` | `0.0.60` / `5.2.1` / `6.1.3` / `1.15.3` | URDF 解析与网格加载校验 |
| `av` | `15.1.0` | PyAV 视频解码/编码校验（Windows 下固定使用 `video_backend="pyav"`，无需 `torchcodec`） |
| `numpy` / `pandas` / `pyarrow` / `datasets` | `2.2.6` / `2.3.3` / `22.0.0` / `4.1.1` | 向量计算、Parquet 与 HuggingFace Arrow 序列化 |
| `zstandard` / `protobuf` | `0.25.0` / `6.33.2` | `.SYNC.mcap` zstd 块解压与 Protobuf 描述池解析 |
| `torch` / `torchvision` | `2.7.1` / `0.22.1` | `lerobot==0.4.4` 与 `LeRobotDataset` 核心运行时 |
| `ffmpeg` / `ffprobe` | `8.0.1` (`8.0.1-full_build`) | 需包含 `libx265`（支持 `hevc_amf` / `hevc_nvenc` / `hevc_qsv` 硬件编码或 `libx265`）并配置到系统 `PATH` |

### 新环境初始化命令

方式一：通过 Conda 环境文件一键初始化：

```powershell
conda env create -f lerobot_converter/environment.yml
conda activate lerobot
```

方式二：手动创建 Conda 环境并用 `pip` 精确安装：

```powershell
conda create -n lerobot python=3.10.19 -y
conda activate lerobot
pip install -r lerobot_converter/requirements.txt
```

安装完成后检查 `ffmpeg`、`ffprobe` 与核心依赖导入是否正常：

```powershell
ffmpeg -version
ffprobe -version
python -c "import lerobot, yourdfpy, av, numpy, pandas, pyarrow, datasets, zstandard, google.protobuf; print(lerobot.__version__, yourdfpy.__version__, av.__version__, numpy.__version__, pandas.__version__, pyarrow.__version__, datasets.__version__, zstandard.__version__, google.protobuf.__version__)"
```

## 脚本参数变量集中维护

所有运行参数均集中在 [config.py](config.py) 中维护，直接编辑该 Python 脚本即可控制全流程（同时保留 `--video-source` 等可选命令行覆盖参数，并在 [convert_to_lerobot.py](convert_to_lerobot.py) / [validate_lerobot.py](validate_lerobot.py) 顶部引用）：

- **路径与任务清单**（`INPUT_ROOT`、`OUTPUT_ROOT`、`URDF_DESC_ROOT`、`TASK_MAPPINGS`）：统一配置原始采集目录、输出目录、URDF 描述目录及各任务的 `TaskMappingConfig`。
- **视频来源开关**（`VIDEO_SOURCE` / `--video-source`）：
  - `mp4`（默认）：直接读取轨迹目录内现成的 `camera_*.mp4`，4 路相机并发经 `ffmpeg` 转码为 `640x480` HEVC（`yuv420p`、`GOP=8`、`bf=0`、无音频），并并发预解析各集的 `.SYNC.mcap`。
  - `mcap`：通过 `parse_sync_mcap(..., dst_video_paths=...)` 在单遍读取 `.SYNC.mcap` 的同时解析非视频流，并将 4 路 `CompressedImage` JPEG 帧通过 `image2pipe` 管道并发直出编码为合规 HEVC 视频（不依赖目录内 `.mp4`）。
- **非视频状态数据来源开关**（`STATE_SOURCE` / `--state-source`）：
  - `mcap`（默认）：优先从 `.SYNC.mcap` 单遍解析 `singorix/wbcs/sensor`（关节）与 `/odom/base_link`（里程计），省去解析大体积 JSON；若 MCAP 内缺少对应流则自动回退读取 `joints.json` / `odom_base_link.json`。
  - `json`：强制从 `joints.json` 与 `odom_base_link.json` 读取状态数据。
- **HEVC 编码与性能/验收参数**：
  - `VIDEO_CODEC`（默认 `auto`）与 `X265_PRESET`（默认 `superfast`）：自动按 `hevc_nvenc -> hevc_amf -> hevc_qsv` 试编码探测硬件编码器并回退 `libx265`；其余质量参数（`VIDEO_CRF_OR_QP`、`VIDEO_GOP_SIZE`、`VIDEO_MAX_B_FRAMES`）、并发线程数（`MCAP_PREFETCH_MAX_WORKERS`、`CAMERA_ENCODE_WORKERS`、`MESH_COPY_WORKERS`、`VALIDATE_VIDEO_MAX_WORKERS`）及验收默认路径/阈值（`VALIDATE_ROOT_DIR`、`PTS_TOLERANCE_S` 等）均在 [config.py](config.py) 内直接维护。

## 使用方法

### 1. 执行转换

默认配置（视频读现有 `mp4`，非视频状态优先读 `mcap`，自动探测硬件 HEVC 编码器）：

```powershell
D:\software\miniconda3\envs\lerobot\python.exe -m lerobot_converter.convert_to_lerobot
```

切换为纯 MCAP 图像流转码（视频与状态均来自 `.SYNC.mcap`）：

```powershell
D:\software\miniconda3\envs\lerobot\python.exe -m lerobot_converter.convert_to_lerobot --video-source mcap --state-source mcap
```

强制使用软件编码与指定预设：

```powershell
D:\software\miniconda3\envs\lerobot\python.exe -m lerobot_converter.convert_to_lerobot --vcodec libx265 --x265-preset superfast
```

自定义输入输出路径：

```powershell
D:\software\miniconda3\envs\lerobot\python.exe -m lerobot_converter.convert_to_lerobot `
  --input-root f:\data-sample `
  --output-root f:\data-sample\lerobot_output `
  --urdf-desc-root F:\config\galbot_description\galbot_one_golf_description
```

### 2. 执行独立验证

验证输出总目录下的全部任务子集：

```powershell
D:\software\miniconda3\envs\lerobot\python.exe -m lerobot_converter.validate_lerobot --root-dir f:\data-sample\lerobot_output
```

验证单个数据集并刷新 `meta/agibot/quality.json` 与 `quality.md`：

```powershell
D:\software\miniconda3\envs\lerobot\python.exe -m lerobot_converter.validate_lerobot --dataset-dir f:\data-sample\lerobot_output\task_flower_arrangement --write-report
```

## 状态与动作向量契约

- `observation.state` 与 `action` 均为 68 维 `float32` 扁平向量，严格同构（`action[t]` 对应下一帧目标状态 `state[t+1]`），完整索引由 `meta/info.json` 的 `field_descriptions` 定义：
  - `0..13`：`left_arm` / `right_arm` 关节角（rad）
  - `14..27`：`left_arm` / `right_arm` 关节角速度（rad/s）
  - `28..41`：`left_end` / `right_end` 末端位置（m）与四元数 `[qx, qy, qz, qw]`（URDF 正运动学计算于 `base_link`）
  - `42..47`：`left_effector` / `right_effector` 的 `position`（m）、`ratio`（`[0, 1]`）与 `width`（m，`[0, 0.12]`）
  - `48..51`：`head` 关节角与速度
  - `52..61`：`leg` 关节角与速度
  - `62..67`：`chassis` 里程计平面位置 `[x, y]` 与归一化四元数 `[qx, qy, qz, qw]`
  - 推荐未采集进入 68 维向量的字段 `state/end/wrench` 与 `action/end/wrench` 已按规范 2.3/4.2 在 `field_descriptions` 中显式声明为 `dimensions=0, indices=[]`
- 标量/索引列：`episode_index`、`frame_index`、`index`、`task_index`、`ts`（int64 纳秒绝对时间）、`timestamp`（float32 相对秒）、`action_src_status`（`1.0`）。

## I/O 与转换效率优化

- **单遍解析与并发预解析**：`mp4` 模式下使用线程池并发预解析全部轨迹的 `.SYNC.mcap` 并跳过无效视频流负载；`mcap` 模式下在同一遍扫描中并行向 4 个 `ffmpeg` 子进程写入 JPEG 帧并提取状态数据，避免重复解压数 GB 的 `.SYNC.mcap`。
- **硬件 HEVC 自动探测与 4 路并发编码**：启动时自动探测 `hevc_nvenc`/`hevc_amf`/`hevc_qsv` 并在失败时回退 `libx265 -preset superfast`，每集 4 路相机并发转码。
- **关键帧采样像素统计**：转换末尾统计视频均值/方差/分位数时使用 PyAV `skip_frame="NONKEY"` 跳过非关键帧，大幅缩短解码耗时。
- **零重复读盘**：URDF 网格在多线程分块拷贝的同时流式计算 SHA256；`camera_params.json` 采用紧凑格式写入。
- **单遍并发验收**：`validate_lerobot.py` 使用 PyAV 单遍同时完成编码参数、PTS 容差与 GOP 结构校验，并以线程池并发验收视频与 URDF 网格哈希。
