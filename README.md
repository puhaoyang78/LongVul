# LongVul

PrimeVul_original 长函数漏洞检测基线。实验采用 RGCL 的原始配对划分，Train、Valid、Test 分别有 8,708、1,124、1,128 个函数。每个函数独立输入和分类。采用种子 42、验证集 MCC 选模（并列取最早轮）和固定判定规则 `p > 0.5`。分层使用未经截断的 CodeBERT 正文 BPE 数。

## VulnScout 与 SEA

### 实现与任务适配

- **VulnScout。** 复用官方 MLP（768→512→256→1，BatchNorm、ReLU、dropout 0.3）和 adaptive focal loss（γ=2）。与官方 tokenizer 调用一致，输入最长 512 含 CLS 和 SEP，`stride=256` 表示重叠 256 个正文 BPE，因此相邻窗口前移 254。冻结 CodeBERT，先对各窗口最后一层有效 token 做均值，再对同一函数的窗口向量等权平均。用单卡批量编码并缓存训练、验证的函数向量。保留官方逆频率有放回抽样、batch 64、Adam（学习率 1e-4，权重衰减 1e-5）、余弦衰减（最低 1e-6）、最多 100 轮和耐心值 10。配对划分的正负样本等量，实际抽样概率等权，focal loss 的 α 为 0.5。早停与选模均用验证 MCC。[IST 论文](https://doi.org/10.1016/j.infsof.2026.108228) · [官方实现](https://github.com/MuhammadFarhatUllah/VulnScout/tree/2834486c1750fe800ea875311b3fa403265bae4f/Experiment)
- **SEA Token Splitting。** 采用论文表 4 的窗口 256、步长 128。README 对应小节的脚本变量写为 32/16，本实验采用该节标题和论文表的配置。上游 Token Splitting 先解析得到词序列，再逐段做 BPE。本实验直接用完整函数的 CodeBERT BPE 构造分窗序列，复用 `sliding_window` 与 `Attention_1`，保持 CodeBERT pooler 输出和官方单层 Attention + Mean。每段在官方 `256+20+16` 的输入预算内打包。每个函数以原前缀视图和分窗为候选，只有一个分窗时沿用原前缀。训练时超过六个候选则有放回地抽六个。验证、测试编码所有候选，并按作者 `predict()` 的行为加入第二份前缀。官方 Mean 项在对窗口求和前广播，所以其贡献随窗口数增长，本实验沿用该计算。将代码搜索的查询与排序损失替换为一个函数级线性 logit 和 BCE。端到端训练 15 轮，有效 batch 48，micro-batch 8，学习率 2e-5，按上游 Transformers AdamW 规则更新，线性调度预热 10 步，梯度裁剪 1。[LREC-COLING 论文](https://aclanthology.org/2024.lrec-main.1347/) · [官方训练入口](https://github.com/fly-dragon211/SEA/blob/8bb61b1baabeb5da3b46005380d6cabdeda53a5c/train_LongCode.py) · [官方注意力实现](https://github.com/fly-dragon211/SEA/blob/8bb61b1baabeb5da3b46005380d6cabdeda53a5c/Model/Attention.py)

单张 RTX 4090 上使用 FP32 和 PyTorch SDPA。每次推理最多向 CodeBERT 提交 64 个窗口。SEA 训练的单个 micro-batch 最多 48 个窗口。每个函数的聚合器收到该函数的全部已编码窗口，窗口批大小只影响送入 CodeBERT 的次数。预检用训练集中最长的真实函数验证全窗口覆盖、有限输出与批次数，并执行相应训练方式的反向和优化器更新。

### PrimeVul_original 测试结果

逐行核对五种方法的验证和测试成员、标签及原始长度。PR-AUC 为 average precision。历史 Head 和 Uniform-4 的代码与逐样本预测来自已发布提交。[历史结果](https://github.com/puhaoyang78/LongVul/tree/e094c1ef8bd5e56b08f154a416f89a9eb9c3ecf6/results/paired)

| 模型 | 整体 MCC | F1 | ROC-AUC | PR-AUC | 长函数 MCC | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CodeBERT Head | 0.0035 | 0.6510 | 0.5104 | 0.5147 | -0.0100 | 0.6348 | 0.4947 | 0.4880 |
| Uniform-4 | 0.1015 | 0.5275 | 0.5559 | 0.5522 | 0.0967 | 0.4993 | 0.5490 | 0.5282 |
| StagedVulBERT | 0.0904 | 0.5448 | 0.5599 | 0.5598 | 0.0626 | 0.4954 | 0.5416 | 0.5344 |
| VulnScout | 0.0108 | 0.4529 | 0.5190 | 0.5315 | 0.0010 | 0.4149 | 0.5134 | 0.5093 |
| SEA | 0.0887 | 0.4926 | 0.5521 | 0.5341 | 0.0618 | 0.4380 | 0.5397 | 0.5179 |

长函数为正文超过 510 BPE 的 823 个测试函数，整体为 1,128 个。VulnScout 训练 24 轮触发早停，第 14 轮入选（Valid MCC 0.0505）。SEA 训练满 15 轮，第 8 轮入选（Valid MCC 0.1063）。

SEA 的整体和长函数 MCC 接近 StagedVulBERT。在 ≥2,041 BPE 组，SEA 的 MCC 为 0.0339，介于 StagedVulBERT 的 0.0044 和 Uniform-4 的 0.0660 之间。VulnScout 在该组的 MCC 为 -0.0244。两种分窗方法都覆盖了全部测试函数，Uniform-4 的整体和长函数 MCC 仍最高。

| 原始正文 BPE | 函数 | 模型 | MCC | F1 | ROC-AUC | PR-AUC |
|---|---:|---|---:|---:|---:|---:|
| ≤510 | 305 | CodeBERT Head | 0.0853 | 0.6911 | 0.5211 | 0.5427 |
| ≤510 | 305 | Uniform-4 | 0.1029 | 0.5904 | 0.5692 | 0.5986 |
| ≤510 | 305 | StagedVulBERT | 0.1532 | 0.6484 | 0.5914 | 0.6079 |
| ≤510 | 305 | VulnScout | 0.0203 | 0.5358 | 0.5250 | 0.5780 |
| ≤510 | 305 | SEA | 0.1430 | 0.6061 | 0.5956 | 0.6030 |
| 511–1,020 | 242 | CodeBERT Head | -0.0025 | 0.6433 | 0.4943 | 0.5110 |
| 511–1,020 | 242 | Uniform-4 | 0.1154 | 0.5368 | 0.5707 | 0.5477 |
| 511–1,020 | 242 | StagedVulBERT | 0.1187 | 0.5900 | 0.5821 | 0.5718 |
| 511–1,020 | 242 | VulnScout | 0.0326 | 0.4979 | 0.5280 | 0.5390 |
| 511–1,020 | 242 | SEA | 0.0907 | 0.5089 | 0.5642 | 0.5459 |
| 1,021–2,040 | 235 | CodeBERT Head | -0.0065 | 0.6364 | 0.4895 | 0.4802 |
| 1,021–2,040 | 235 | Uniform-4 | 0.1218 | 0.4976 | 0.5564 | 0.5520 |
| 1,021–2,040 | 235 | StagedVulBERT | 0.0872 | 0.5023 | 0.5274 | 0.5332 |
| 1,021–2,040 | 235 | VulnScout | -0.0015 | 0.4179 | 0.5146 | 0.5102 |
| 1,021–2,040 | 235 | SEA | 0.0693 | 0.4293 | 0.5419 | 0.5266 |
| ≥2,041 | 346 | CodeBERT Head | -0.0174 | 0.6276 | 0.4938 | 0.4841 |
| ≥2,041 | 346 | Uniform-4 | 0.0660 | 0.4721 | 0.5280 | 0.5100 |
| ≥2,041 | 346 | StagedVulBERT | 0.0044 | 0.4042 | 0.5187 | 0.5084 |
| ≥2,041 | 346 | VulnScout | -0.0244 | 0.3396 | 0.5010 | 0.4878 |
| ≥2,041 | 346 | SEA | 0.0339 | 0.3852 | 0.5208 | 0.5083 |

### 实际覆盖和成本

| 模型 | 测试完整函数 | 测试正文 BPE 覆盖 | 测试 PLM 窗口 | 训练（分钟） | Test 推理（秒） | 峰值显存（GiB，训练 / 推理） |
|---|---:|---:|---:|---:|---:|---|
| CodeBERT Head | 305 / 1,128 | 16.37% | 1,128 | 44.97 | 4.71 | 历史未记录 |
| Uniform-4 | 305 / 1,128 | 15.53% | 1,128 | 36.54 | 4.75 | 历史未记录 |
| StagedVulBERT | 785 / 1,128 | 43.16% | 2,910（TE） | 88.26 | 21.51 | 12.71 / 1.36 |
| VulnScout | 1,128 / 1,128 | 100.00% | 11,660 | 4.77 | 35.93 | 1.68 / 1.68 |
| SEA | 1,128 / 1,128 | 100.00% | 26,784 | 63.11 | 46.52 | 10.37 / 2.03 |

正文覆盖按去除重复窗口后的总 BPE 数计算。StagedVulBERT 的分母为其逐行分词的正文 BPE，TE 窗口数之外，每个函数还有语句级编码。训练与推理计时分别排除验证和分词准备。显存为 PyTorch 峰值已分配字节。原有三组使用 FP32 eager，本次两组使用 FP32 SDPA。

VulnScout 的训练时间包括一次训练特征提取（268.77 秒）和 24 轮 MLP 优化（17.73 秒）。SEA 用训练期间实际抽取的窗口覆盖了 29.93% 的正文 BPE 观测，分母按函数被训练的次数累加。两种方法在测试时完整覆盖最长的 77,547 BPE 函数。

测试集的 CodeBERT 批调用次数分别为 VulnScout 190 和 SEA 425，最长函数分别编码 305 和 608 个窗口。[VulnScout 原始结果](results/vulnscout/) · [SEA 原始结果](results/sea/) · [最佳模型](https://github.com/puhaoyang78/LongVul/releases/tag/window-baselines-primevul-seed42)


### 复现

在本仓库目录执行，路径与下方 StagedVulBERT 相同。源代码经服务器 Mihomo 的 127.0.0.1:7890 获取，模型读取本地 CodeBERT。若在独立副本重跑，先移走两组已发布的结果目录。模型与训练恢复状态存储在 `checkpoints/<method>/`。Release 中的 VulnScout 权重为官方 MLP 的 state dict，与本地冻结 CodeBERT 合用。SEA 权重含 CodeBERT、聚合器和分类头。

```bash
git -c http.proxy=http://127.0.0.1:7890 submodule update --init
conda activate vul-detect
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=true
tmux new -s longvul-window42
python -u window_baselines.py vulnscout && python -u window_baselines.py sea
```

`train_coverage.csv.gz` 保存逐函数训练覆盖和 PLM 前向计数。VulnScout 计一次冻结编码，SEA 计所有训练轮次。Valid/Test CSV 还包含概率和阈值判定。`plm_windows` 是每个函数送入 PLM 的窗口数，包含重复候选。`plm_batch_calls` 是包含该函数窗口的调用次数，共享批次可能包含多个函数。汇总 JSON 的批调用数对共享批次只计一次。覆盖按实际输入跨度的并集计数，每个正文 BPE 最多计一次。

## StagedVulBERT

本仓库将 StagedVulBERT 的函数级分类器迁移到 PrimeVul_original。TETransformer 和 SETransformer 各 6 层，保留作者的 Token-to-Statement 聚合、两类 softmax 分类头和函数级交叉熵。模型与特征转换直接调用固定提交的上游代码。[TSE 论文](https://doi.org/10.1109/TSE.2024.3493245) · [官方函数级入口](https://github.com/YuanJiangGit/StagedVulBERT/blob/d589175880ff99b0192f4ded54f5455951a696fe/Entry/StagedBert_vul.py) · [模型实现](https://github.com/YuanJiangGit/StagedVulBERT/blob/d589175880ff99b0192f4ded54f5455951a696fe/Models/StagedModel_vul.py)

### PrimeVul_original 测试结果

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

### 数据与模型初始化

通过相邻 RGCL 仓库的 `prime_source_rows` 读取官方配对划分，Train / Valid / Test 分别为 8,708 / 1,124 / 1,128 个函数，训练随机打散函数。数据加载器只向模型传入本函数的输入和类别。分层使用完整原函数的 CodeBERT BPE 长度；模型的覆盖量使用作者逐行分词后的正文 BPE，这两个口径分别记录在逐样本预测中。

初始化文件 `pretrain_model.bin` 来自作者的 MSP 预训练，217 个归档成员均通过 CRC 检查。加载后逐项核对所有匹配张量的值、形状、类型和有限性。两级各加载 105 个张量，共 164,238,338 个元素。MSP 专用的 Q、K 和 7 个解码头张量不参与下游前向。下游缺少来源的 105 项中，4 项是新函数分类头，101 项是官方包装类继承的顶层 RoBERTa，后者不参与前向。模型实际使用两级模块各自的 RoBERTa。作者另行发布的 `model_2048.bin` 是微调模型，实验使用 MSP 文件初始化。[逐键加载核对](results/stagedvulbert/initialization.json)

### 训练与评价

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

#### 分段口径

作者逐行分词，空行以换行符表示。先按 `min(4, ceil((正文 BPE 数 + 1)/512))` 确定段数，各段前加 CLS，尽可能在行边界收尾。遇到超长单行时，原实现可能反复停在同一行起点，产生只有 CLS 的段。结果分别统计名义上需要超过四段、实际有正文未覆盖、四段仍未覆盖，以及整函数零正文的数量。所有样本均进入评价分母。[正式实现](stagedvulbert.py)

### 复现

在 GPU-SERVER-3 使用相邻的 RGCL 仓库（提交 `d8570b98fabe92caf15bcc075ac7796ef307b3b3`）、`../../data/PrimeVul_original` 与 `../../../resource/{codebert-base,StagedVulBERT}`。本次权重从本地加载，官方源码经服务器 Mihomo 的 127.0.0.1:7890 取得。

```bash
git -c http.proxy=http://127.0.0.1:7890 submodule update --init
conda activate vul-detect
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -u stagedvulbert.py
```

入口在正式训练前自动做优化器测试；只执行测试可加 `--smoke-only`。在已有发布结果的独立复现副本中先移走 `results/stagedvulbert/`。最佳模型保存为 `checkpoints/stagedvulbert/best.pt`。逐轮指标、预测、配置与覆盖统计见[结果目录](results/stagedvulbert/)。
