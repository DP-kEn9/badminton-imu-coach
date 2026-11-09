# 测试报告 · 2026-11-09

Windows / Python 3.12.14 / CPU；pytest：**34 passed in 9.05s**，0 failed、0 skipped；ruff：All checks passed。
所有测试用例只用原创 fixture，不依赖受限资产、个人密钥或付费 API。

| 实际命令（项目虚拟环境） | 退出码 | 耗时 s |
|---|---:|---:|
| `python -m badminton_coach.cli doctor` | 0 | 3.871 |
| `python -m pytest -q --basetemp .tmp/pytest-1794194969089828500` | 0 | 10.604 |
| `python -m ruff check .` | 0 | 0.179 |
| `python -m badminton_coach.cli evaluate --run-dir artifacts/smoke-v02` | 0 | 5.730 |
| `python -m badminton_coach.cli report --run-dir artifacts/smoke-v02 --backend template` | 0 | 3.989 |

另外实际运行成功（退出码 0）：

- `python -m badminton_coach.cli demo --config configs/smoke.yaml --device cpu --output artifacts/smoke-v02`：60 步，14.764 s（训练模块计时，不含完整评测）。
- `python -m badminton_coach.cli train --config configs/smoke.yaml --device cpu`：60 步，19.509 s。
- `python -m pip install -e . --no-deps --no-build-isolation`：成功；实际安装版本已冻结 requirements.lock.txt。

覆盖 FK/骨长/绑定、静止/平移/旋转、同刚体加速度差、退化 6D、时钟/长短缺口/饱和/安装、VQF 静止/连续航向/符号、非静止校准拒绝、缺节点保留直接分析、错误单位、无第五节点、先分组后切窗、短序列/尾窗、证据校验/降级/replay/秘密不记录、授权 fixture/unsupported、真实反传/重载/GT 隔离、CPU 端到端。

首次测试 23 passed / 1 failed / 10 errors：同刚体加速度差的二阶差分截断误差超容差；未放宽容差，改为五点中心二阶差分后通过。10 个 errors 为默认系统临时目录写入权限，改用项目 .tmp 后通过。早期 pip 网络失败后使用受控网络安装成功；Python 3.11 未找到，改为项目 Python 3.12。
最终复验在数值修正后生成新的 smoke-v02，不覆盖旧 demo。完整本地核验记录 artifacts/verification.json；早期错误不计作最终通过。

未运行：匹配 oracle 训练、多种子、真实外部动作/人体模型、真实硬件、live LLM、远程 CI、图形界面播放视觉验收。它们不是 passed，也没有伪记为 pytest skip。
HTML 的生成、数据内嵌、预测/GT 文件隔离已通过自动测试；视觉和交互运行未独立验收。
