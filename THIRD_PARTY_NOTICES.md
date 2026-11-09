# 第三方依赖和授权

原创解析骨架与参数化动作不来自外部人体模型。没有下载 AMASS、SMPL、预训练人体权重或受限动作。本项目未自动选择开源许可证。

实际安装版本见 requirements.lock.txt。安装包自带完整许可文本；以下为依赖许可摘要，不替代其文本与传递依赖要求。

| 依赖 | 许可 |
|---|---|
| NumPy / SciPy | BSD-3-Clause |
| PyTorch | BSD-3-Clause |
| VQF | MIT |
| PyYAML / pytest / ruff | MIT |
| setuptools | MIT |

传递依赖包括 filelock (Unlicense)、fsspec (BSD-3-Clause)、networkx (BSD-3-Clause)、Jinja2/MarkupSafe (BSD-3-Clause)、mpmath (BSD-3-Clause)、SymPy (BSD-3-Clause)、typing_extensions (PSF)、packaging (Apache-2.0/BSD)、pluggy (MIT)、iniconfig (MIT)、Pygments (BSD-2-Clause)、colorama (BSD-3-Clause)。以安装元数据和 LICENSE 文件为准；产品分发前复核 notices。

方向融合参考 [VQF 官方 offlineVQF 接口](https://vqf.readthedocs.io/en/stable/ref_cython_offline.html)：等采样率、连续数组、角速度 rad/s、比力 m/s²、无磁场输入。
LLM HTTP 适配按 [OpenAI 官方 Chat Completions 协议](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create) 编写；未固定模型名称、未发 live 付费请求。

configs/license_manifest.json 维护数据、模型、代码和权重分项授权。私有仓库不能代替第三方授权。凭证引用指向本地保管，不提交授权原件或个人数据。
