# 羽毛球智能教练 · 阶段 I smoke

四身体 IMU 离线研究原型。使用原创 21 节点解析骨架，完成原始六轴合成、VQF 融合、真实 BiGRU 训练、独立预测、FK、HTML 回放和受证据约束的报告。球拍 `null`，没有第五路观测。

当前交付是 **smoke 工程验证**。所有样例均为 `synthetic_analytic / synthetic_demo_only`；授权动作、真实硬件和产品能力均 `not_validated`。未部署服务，也不启动后台训练。

项目仓库：[DP-kEn9/badminton-imu-coach](https://github.com/DP-kEn9/badminton-imu-coach)。仓库公开上传已经项目所有者明确授权；未选择开源许可证。

```powershell
git clone https://github.com/DP-kEn9/badminton-imu-coach.git
cd badminton-imu-coach
```

## 安装与运行

Python 3.11+；本机实际验证 Windows / Python 3.12.14 / CPU。使用项目虚拟环境，不修改系统 Python。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m badminton_coach.cli doctor
python -m badminton_coach.cli demo --config configs/smoke.yaml --device cpu --output artifacts/my-smoke
python -m pytest -q
python -m ruff check .
```

本机已有 `.venv`，也可直接用 `.\.venv\Scripts\python.exe` 替代 `python`。若 Windows 临时目录被限制，先将 `TEMP`、`TMP` 指向项目内 `.tmp`，测试加 `--basetemp .tmp/pytest-new-id`。输出目录已存在时拒绝覆盖，换新的 run ID。

```powershell
python -m badminton_coach.cli train --config configs/smoke.yaml --device cpu --output artifacts/train-new
python -m badminton_coach.cli evaluate --run-dir artifacts/smoke-v02
python -m badminton_coach.cli report --run-dir artifacts/smoke-v02 --backend template
```

打开 `artifacts/smoke-v02/replay.html` 播放预测，暂停和拖动时间条；`evaluation_replay.html` 是单独的研发 GT 对照。两个文件均内嵌数据，不依赖 CDN、GUI 框架或 ffmpeg。当前三维坐标采用固定斜投影，未实现自由相机。

产物含 `checkpoint.pt`、`prediction.npz/json`、`coach_report.md/json`、`evaluation.md/json`、`training.json`、`provenance.json`，以及分层原始/融合数据。产物仅保存在本地，不提交 Git。

## 合同和边界

身体顺序 pelvis / left_shank / right_shank / racket_forearm；持拍侧由元数据决定。W 系 X 前 Y 左 Z 上；旋转 S→W，外部四元数 wxyz；m、s、rad/s、m/s²。输出骨盆系相对姿态，根平移为空。六轴融合没有绝对航向。

`InferenceSession` 不包含 GT 路径；`EvaluationTruth` 只由训练和 BenchmarkRunner 使用。直接特征来自原始有效且未饱和帧；比力 RMS 含重力，角速度峰值不命名为击球或发力起点。两条分析通路同源，不是独立验证。

`assets.load_licensed` 实现经过许可清单与哈希检查的 `analytic_npz_v1 + analytic_skeleton_v1` 加载组合，仅用原创 fixture 验证。SMPL/AMASS 等返回 unsupported；授权动作到训练管线的完整集成尚未验证。不得截断或套用不同拓扑权重。

报告后端支持 template / mock / replay / live；live 必须显式设置 `.env.example` 中配置，程序不会自动读取 `.env`。只发送结构化证据；LLM 只能选择证据 ID，数值由程序渲染；失败降级模板。没有进行付费 live 请求。

设计文档：[阶段规划](docs/01_产品开发阶段规划.md)、[功能需求](docs/02_第一阶段实现功能需求.md)、[技术路线](docs/03_第一阶段技术路线.md)。

交付记录：[实施报告](IMPLEMENTATION_REPORT.md)、[测试报告](TEST_REPORT.md)、[已知限制](KNOWN_LIMITATIONS.md)、[下一步](NEXT_STEPS.md)。
