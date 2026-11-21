# LongVul · StagedVulBERT

本仓库将 StagedVulBERT 的函数级分类器迁移到 PrimeVul_original。TETransformer 和 SETransformer 各 6 层，保留作者的 Token-to-Statement 聚合、两类 softmax 分类头和函数级交叉熵。模型与特征转换直接调用固定提交的上游代码。[TSE 论文](https://doi.org/10.1109/TSE.2024.3493245) · [官方函数级入口](https://github.com/YuanJiangGit/StagedVulBERT/blob/d589175880ff99b0192f4ded54f5455951a696fe/Entry/StagedBert_vul.py) · [模型实现](https://github.com/YuanJiangGit/StagedVulBERT/blob/d589175880ff99b0192f4ded54f5455951a696fe/Models/StagedModel_vul.py)

## PrimeVul_original 测试结果

验证 MCC 在第 14 轮最高（0.1125）。表中均为种子 42 的 PrimeVul 结果，PR-AUC 使用 average precision，长短按原始正文 BPE 判定。历史 Head、Uniform-4 与本次的 Valid/Test 成员、标签和原始长度逐行相同；两组历史基线使用有效 batch 8、10% 预热和 LongVul 的 LOSVERAdamW。[历史预测与指标](https://github.com/puhaoyang78/LongVul/tree/e094c1ef8bd5e56b08f154a416f89a9eb9c3ecf6/results/paired)

| 模型 | 测试范围 | 函数 | MCC | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CodeBERT Head | 整体 | 1128 | 0.0035 | 0.6510 | 0.5005 | 0.9309 | 0.5104 | 0.5147 |
| CodeBERT Head | >510 | 823 | -0.0100 | 0.6348 | 0.4893 | 0.9035 | 0.4947 | 0.4880 |
| CodeBERT Head | 2041+ | 346 | -0.0174 | 0.6276 | 0.4854 | 0.8876 | 0.4938 | 0.4841 |
| Uniform-4 | 整体 | 1128 | 0.1015 | 0.5275 | 0.5560 | 0.5018 | 0.5559 | 0.5522 |
| Uniform-4 | >510 | 823 | 0.0967 | 0.4993 | 0.5490 | 0.4579 | 0.5490 | 0.5282 |
| Uniform-4 | 2041+ | 346 | 0.0660 | 0.4721 | 0.5294 | 0.4260 | 0.5280 | 0.5100 |
| StagedVulBERT | 整体 | 1128 | 0.0904 | 0.5448 | 0.5453 | 0.5443 | 0.5599 | 0.5598 |
| StagedVulBERT | >510 | 823 | 0.0626 | 0.4954 | 0.5265 | 0.4678 | 0.5416 | 0.5344 |
| StagedVulBERT | ≤510 | 305 | 0.1532 | 0.6484 | 0.5784 | 0.7375 | 0.5914 | 0.6079 |
| StagedVulBERT | 511-1020 | 242 | 0.1187 | 0.5900 | 0.5461 | 0.6417 | 0.5821 | 0.5718 |
| StagedVulBERT | 1021-2040 | 235 | 0.0872 | 0.5023 | 0.5400 | 0.4696 | 0.5274 | 0.5332 |
| StagedVulBERT | 2041+ | 346 | 0.0044 | 0.4042 | 0.4915 | 0.3432 | 0.5187 | 0.5084 |

StagedVulBERT 整体 MCC 高于 Head，低于 Uniform-4；整体 F1、ROC-AUC 与 PR-AUC 均略高于 Uniform-4。随着长度从 511–1,020 增至 ≥2,041 BPE，StagedVulBERT 的 MCC 从 0.1187 降至 0.0044。最长一档的四段覆盖仍有限；这组实验不能将性能差异单独归因于覆盖范围。

| 划分 | 最多实际覆盖正文 BPE | 名义需要超过四段 | 仍有正文被截断 | 用满四段仍被截断 |
|---|---:|---:|---:|---:|
| Train | 2,038 | 2,419 | 2,562 / 8,708 | 2,475 |
| Valid | 2,035 | 300 | 319 / 1,124 | 307 |
| Test | 2,036 | 328 | 343 / 1,128 | 335 |

测试集中另有 8 个函数在少于四段时就出现未覆盖正文，这是上游预估段数与实际行边界共同造成的。训练集中 2 个函数出现空正文段；三份划分均无整个函数零正文的情况。

| 方法 | 16 轮训练（分钟） | 整个 Test 推理（秒） | 训练峰值（GiB） |
|---|---:|---:|---:|
| CodeBERT Head | 44.97 | 4.71 | 历史未记录 |
| Uniform-4 | 36.54 | 4.75 | 历史未记录；同配置两步实测 6.415 |
| StagedVulBERT | 88.26 | 21.51 | 12.71 |

StagedVulBERT 测试推理峰值 1.36 GiB；相较 Uniform-4，训练耗时约 2.42 倍，推理约 4.53 倍。可下载的最佳权重为官方 Model 的完整 state dict，已严格重载并复现第 14 轮完整验证指标。[最佳模型](https://github.com/puhaoyang78/LongVul/releases/tag/stagedvulbert-primevul-seed42) · [逐轮历史](results/stagedvulbert/epochs.json) · [测试指标](results/stagedvulbert/metrics.json) · [逐样本测试预测](results/stagedvulbert/test.csv.gz)

## 数据与模型初始化

通过相邻 RGCL 仓库的 `prime_source_rows` 读取官方配对划分，Train / Valid / Test 分别为 8,708 / 1,124 / 1,128 个函数，训练随机打散函数。数据加载器只向模型传入本函数的输入和类别。分层使用完整原函数的 CodeBERT BPE 长度；模型的覆盖量使用作者逐行分词后的正文 BPE，这两个口径分别记录在逐样本预测中。

初始化文件 `pretrain_model.bin` 来自作者的 MSP 预训练，217 个归档成员均通过 CRC 检查。加载后逐项核对所有匹配张量的值、形状、类型和有限性。两级各加载 105 个张量，共 164,238,338 个元素。MSP 专用的 Q、K 和 7 个解码头张量不参与下游前向。下游缺少来源的 105 项中，4 项是新函数分类头，101 项是官方包装类继承的顶层 RoBERTa，后者不参与前向。模型实际使用两级模块各自的 RoBERTa。作者另行发布的 `model_2048.bin` 是微调模型，实验使用 MSP 文件初始化。[逐键加载核对](results/stagedvulbert/initialization.json)

## 训练与评价

| 设置 | 上游函数级默认值 | 本实验 |
|---|---|---|
| 编码 | block_size 512，seg_num 4 | 相同 |
| 优化器 | PyTorch AdamW，2e-5，epsilon 1e-8，权重衰减 0，梯度裁剪 1 | 相同 |
| 调度 | 前 20% 线性预热，随后线性衰减 | 按实际优化器步数计算 |
| 训练 batch | 32 | micro-batch 8，累积 4 次，有效 32，尾批 4 个函数也更新 |
| 种子 / 轮数 | 12345 / 10 | 42 / 16 |
| 选模 | 训练中按验证 F1 | 每轮完整验证 MCC，并列取最早轮 |
| 二分类 | 正类 softmax 概率严格大于 0.5 | 相同 |

单卡 FP32，采用 eager attention 对齐上游 Transformers 4.34 的注意力实现。显存指标为 PyTorch 峰值已分配字节。训练时间累计每轮训练和优化器更新；验证、预处理、权重读写分别计量或排除。推理时间包含数据加载、掩码构造、搬运、两级前向及取回概率。正式训练前在真实长函数上检查前向、两级梯度、分类头梯度及一次参数更新，再重新从 MSP 权重和种子 42 开始训练。

### 分段口径

作者逐行分词，空行以换行符表示。先按 `min(4, ceil((正文 BPE 数 + 1)/512))` 确定段数，各段前加 CLS，尽可能在行边界收尾。遇到超长单行时，原实现可能反复停在同一行起点，产生只有 CLS 的段。结果分别统计名义上需要超过四段、实际有正文未覆盖、四段仍未覆盖，以及整函数零正文的数量。所有样本均进入评价分母。[正式实现](stagedvulbert.py)

## 复现

在 GPU-SERVER-3 使用相邻的 RGCL 仓库（提交 `d8570b98fabe92caf15bcc075ac7796ef307b3b3`）、`../../data/PrimeVul_original` 与 `../../../resource/{codebert-base,StagedVulBERT}`。本次权重从本地加载，官方源码经服务器 Mihomo 的 127.0.0.1:7890 取得。

```bash
git -c http.proxy=http://127.0.0.1:7890 submodule update --init
conda activate vul-detect
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -u stagedvulbert.py
```

入口在正式训练前自动做优化器测试；只执行测试可加 `--smoke-only`。在已有发布结果的独立复现副本中先移走 `results/stagedvulbert/`。最佳模型保存为 `checkpoints/stagedvulbert/best.pt`。逐轮指标、预测、配置与覆盖统计见[结果目录](results/stagedvulbert/)。
