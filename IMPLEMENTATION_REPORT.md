# 实施报告 · 2026-11-09

交付范围按用户最新要求限定为阶段 I **smoke**。未完整部署、未长期训练、未租用 GPU 或新增付费资源。
版本 v0.2-inline；三份设计文档由本次完整任务规范整理，不声明与旧附件一致。

## 实际实现

body / schemas / simulation / fusion / models / pipeline / assets / analysis / agent / visualization / CLI 均为实际模块。
工程选择使用同名平铺模块，减少首版空目录；后续按复杂度拆包。独立 `predict(session, checkpoint)` 只接收推理输入，demo 调用同一训练、预测、报告与评测模块。

Python 3.12.14，PyTorch 2.15.0+cpu，Windows，CPU；未使用 GPU。四节点，无球拍。
数据模式 demo_synthetic；方向模式 fused_orientation；实际后端 VQF offline 六轴。
100 Hz，4 秒会话含 1 秒静止校准，窗口 200、步长 100、双层 BiGRU 每向 hidden 64、batch 8、AdamW 1e-3、clip 1、12 epoch、seed 20261109。
18 个原始配置组（12 train / 3 dev / 3 test），共 60 优化步，耗时 14.764 s；参数变化已验证。
首/末 batch 损失 0.00150338 / 0.00146725，不是研究有效性的证明。
独立 train 命令也运行成功：60 步，19.509 s，输出 artifacts/train。完整内存信息 unavailable（OS 硬件查询拒绝）。无外部 API 或云 GPU 费用；未估算本机能耗。

## F01–F28 状态

implemented / tested 表示本行列明的 smoke 子集，不能解读为整条原始需求研究验收完成。剩余范围明确标出。

| 编号 | 状态 | 实际范围及余项 |
|---|---|---|
| F01 | implemented / tested | 原创骨架与授权 NPZ+JSON 适配 fixture；真实外部集成 not_validated |
| F02 | implemented / tested | 四身体比力和角速度；静止/平移/旋转解析测试 |
| F03 | implemented / tested | knee 小腿 frame 近/远端，刚体站点解析测试 |
| F04 | implemented / tested | 先按原始配置组划分，再切窗；同动作族泛化 not_validated |
| F05 | implemented / tested | 真实反传、60 步、checkpoint 重载、训练集 SO(3) 均值 |
| F06 | deferred | P1 Transformer；按用户要求当前只做 smoke |
| F07 | implemented / tested | 稳定 6D→SO(3)、固定骨长 FK、骨盆相对姿态 |
| F08 | implemented / tested | 独立三序列，逐关节/左右/持拍侧/速度；研究阈值 not_validated |
| F09 | implemented / tested | 自包含播放/暂停/定位与独立 GT 视图；图形视觉验收 blocked_external |
| F10 | implemented / tested | 角速度峰、比力 RMS、有效/饱和率；运动段/事件时差 deferred |
| F11 | implemented / tested | 封闭函数编排与结构化证据协议，配置化 HTTP provider；live not_validated |
| F12 | implemented / tested | ID/单位/数值/时间单调/观察类型检查；异常降级；复杂仲裁 deferred |
| F13 | implemented / tested | 配置、依赖锁、seed、版本、清单/权重哈希、结果日志 |
| F14 | implemented / tested | 组合噪声/偏置/安装/缺帧/错时/饱和；探索幅度目前代码固定，通用配置扩展 deferred |
| F15 | deferred | P1 专项参考时序比较 |
| F16 | implemented / tested | 未知动作/无参考、缺节点、错单位、冲突、工具/网络失败、划分隔离 |
| F17 | deferred | P1 3/2 节点模型；当前缺节点拒绝重建，保留直接分析 |
| F18 | deferred | P2 正式 App/富交互 |
| F19 | deferred | P2 扩散与复杂物理优化 |
| F20 | implemented / tested | 授权准入与哈希/拓扑检查；原创替代数据；外部资产 blocked_external |
| F21 | implemented / tested | 真实 VQF 2.1.2 离线六轴；oracle 为独立评测入口，匹配 oracle 训练 deferred |
| F22 | implemented / tested | 固定输入改 GT 不改预测/证据；默认/已知体型仅评测区分 |
| F23 | implemented / tested | 每节点设备/会话时间、采样率、锚点、校准来源；多采样率重采样 deferred |
| F24 | implemented / tested | 工程 smoke / 研究和产品 not_validated；CPU 步数、耗时；内存查询 unavailable |
| F25 | implemented / tested | CLI、pytest、ruff、项目 venv、本地 Git；CPU CI 未运行，远程 blocked_external |
| F26 | deferred | P1 时变滑移/复杂软组织/硬件拟合 |
| F27 | implemented / tested | mock record/replay、外联禁用与模板降级；live blocked_external |
| F28 | implemented / tested | 中心优先窗口拼接、短序列/尾窗/6D 退化/连续航向与四元数符号 |

## M0–M6 状态

| 里程碑 | 状态 | 范围 |
|---|---|---|
| M0 授权与数据合同 | implemented / tested | 原创数据与授权清单 fixture；真实授权 blocked_external |
| M1 合成与融合 | implemented / tested | 解析信号、VQF、同步/静止校准 smoke |
| M2 BiGRU 与回放 | implemented / tested | 真实 CPU 训练和独立预测；视觉验收未做 |
| M3 布点与鲁棒性 | implemented / tested | E1–E6 小规模结果；完整研究 not_validated |
| M4 直接特征与证据 | implemented / tested | 描述性测量，未知动作，不输出专项诊断 |
| M5 Agent 受控闭环 | implemented / tested | template/mock/replay；live blocked_external |
| M6 研究评审与交付 | implemented / not_validated | smoke 文档与本地代码交付；研究评审和远程 CI 未通过验证 |

## 实验

E1–E6 由同一 evaluator 对三个独立测试配置生成 9 条序列条件结果，完整数值在本地 artifacts/smoke-v02/evaluation.json。
E2 仅做 oracle 输入到 fused 训练模型的跨模式迁移；匹配 oracle 训练未运行。E3 是跨布局。E4 是组合扰动，不能分离各扰动贡献。
E5 已知体型仅在 BenchmarkRunner 用真值骨长做 FK；默认推理不读骨长真值。E6 比较固定根测试中同语义前臂角速度峰，禁止将同源两路当独立验证。
逐关节、左右、持拍侧和速度标签保存在 JSON；测试组只有少量慢速配置，不宣称覆盖快速羽毛球。
未进行大规模、多种子、Transformer、真实人体、专项动作或硬件实验。

## Git 交付（首次本地 smoke 交付记录）

Git 2.47.1 已存在，无需重复安装。已 git init -b main；作者采用用户提供身份，仅设项目局部配置。
无 remote；GitHub CLI 不存在。Edge 虽已登录，自动化因无法可靠确认当前 URL 而被工具停止，未绕过。
创建状态 blocked_external；private 未验证；推送未进行；远程 SHA unavailable；远程 CI not_run。
本地提交 SHA 由交付时 git rev-parse HEAD 确认，不把 init 或 commit 描述为 GitHub 创建成功。

### GitHub 同步续办（2026-11-09）

首次记录之后，用户提供了仓库 https://github.com/DP-kEn9/badminton-imu-coach 并明确授权公开上传。
origin 已设置为该项目地址；GitHub API 已确认 public，当前认证账号具备 push 权限。认证由用户通过本机 Git Credential Manager 完成，凭证未写入项目。
远程分支检查为空，不覆盖已有分支。上传范围为源码、测试、配置模板及文档，包含 README；本地数据、权重、虚拟环境、秘密及完整运行日志排除。
完整提交历史已经扫描；推送 SHA 与 CI 状态以 GitHub 实际结果和本次交付回执为准。首次记录中的 private 约束已由用户后续明确的公开上传授权更新。

## 验收结论

工程 smoke 通过；研究 not_validated；产品 not_validated。授权数据/live LLM/真实硬件均未验证。
缺资源仅阻断对应远程和研究步骤，本地工作已完成。剩余事项与工程差异见 KNOWN_LIMITATIONS.md、NEXT_STEPS.md。
