# 图片前缀逆向模型：首批真实 NPU 数据

采集日期：2026-10-22。本报告记录采集阶段，当时尚未训练逆向模型。
后续已完成训练及图片重建，见 [实验更新](experiments/20261022_npu_prefix/README.md)。

数据目录：[Release 下载](https://github.com/slittlefingers/gemma4-image-prefix-dataset/releases/tag/v2026.10.22-pilot)。
共 8 条长序列、15,816 个目标位置，实际使用 56 张不同图片。
采集程序记录耗时 187.85 秒，不包含开发、构建和后续审查时间。

## 图片、prompt 与运行设备

图片来自已有 `kvidence/datasets/shadow_chat/image_chat_100/media`，其来源为
COCO 2017 validation。原始 100 张图是从至少 3 个标注对象、2 个类别的合格图片池中，
用固定种子 20261019 抽样。此次按种子 20261022 将 100 张图划为 80/10/10 的图片池。
实际使用训练 40 张、验证 8 张、测试 8 张，图片 ID 和文件哈希不跨集合。
每张图的 COCO ID、URL、单独许可、原图哈希、特征哈希与实际分辨率见
[image_manifest.json](metadata/image_manifest.json)。没有下载新图片。

**没有自然语言任务 prompt，也没有 system prompt。** 没有使用原数据配对的 LMSYS 问题，
没有输入 COCO 类别或图片说明。输入只是固定 user-turn 标记与同一轮中的多张图片：

```text
<bos><|turn>user\n\n\n<|image>
[第 1 张图片的 soft embeddings]
[模型的 end-of-image embedding][image-start token]
[第 2 张图片的 soft embeddings]
...
```

这里是结构示意，不是在图片间加入这些中文说明。初始实际 token IDs 为
`[2,105,2364,109,255999]`；后续分隔为真实模型 EOI 常量及 `255999`，不额外插入换行。
最后一张图可能只取一部分位置。每条序列的精确图片顺序、起止位置见
[index.json](metadata/index.json)。这是构造的多图 user turn，不是自然多轮聊天。

设备为 Pixel 10，Gemma 4 E2B Tensor G5 `.litertlm`。
模型 SHA-256：`af1082986639ecde7db95d91be6fe54f8b6b458104734c5bafc204e69d6852dc`。
手机 STB 预处理、CPU vision encoder、CPU adapter、**真实 NPU decoder**。
KV 为第 5 层原始 INT8 输出；不是 MLX 前向结果，也不是模拟量化数据。
采用实际 `vision_280`、`vision_adapter_280`、因果 `prefill_128` 签名。

## 训练对与长度覆盖

对目标位置 p，前缀长度就是其前面绝对 token 位置数量 p，包括模板及此前图片。
采样范围为 10–2000，保留范围内所有图片位置，排除 EOI/BOI 分隔位置。
长前缀由真实多张图片构成，不用随机向量填充。

| 集合 | 序列 | 实际图片数 | 目标位置数 |
|---|---:|---:|---:|
| train | 6 | 40 | 11,862 |
| validation | 1 | 8 | 1,977 |
| test | 1 | 8 | 1,977 |
| 总计 | 8 | 56 | 15,816 |

| 前缀长度 | 目标位置数 |
|---|---:|
| 10–31 | 176 |
| 32–63 | 256 |
| 64–127 | 512 |
| 128–255 | 1,024 |
| 256–511 | 2,032 |
| 512–999 | 3,872 |
| 1000–1499 | 3,968 |
| 1500–2000 | 3,976 |

序列文件共享保存完整前缀。位置 p 的训练输入为 `embeddings[:p]`、`kv_codes[p]` 和 p；
标签为 `embeddings[p]`，不能把当前位置或未来 embedding 放进模型输入。
adapter 后标签为 1536 维；每张图另存 adapter 前 768 维标签，供受 adapter 约束的候选模型使用。
KV 布局是 K 512 维 + V 512 维，量化参数单独保留，支持 LSB 与反量化两种误差分析。

**这批前缀是真实干净 embedding，不是恢复出的 embedding。** 后续需要加入明确标注的噪声增强，
再收集实际恢复轨迹，检验误差积累。训练应按长度分桶采样；相邻位置高度相关，15,816 个位置
不能视为同等数量的独立图片或上下文。

## 检查结果与执行路径限制

- 文件哈希、数组维度、有限数值、图片级划分、逐位置标签对应关系均通过独立验证。
- 手机真实 adapter 重算一致；各图片起始位置的候选查询与写入一致。
- 前 4 个图片位置的逐 token 与批量路径对照完全一致，RMSE 为 0。
- 新建手机 worker，保持原采集分块重放 train_000：**1,982 个图片位置全部逐位一致**。
- 检查实际辅助模型 mask：历史可见范围和当前块因果关系符合预期，局部窗口为 512。
- **改变 prefill 分块或使用单位置候选查询时，部分位置仍有差异。** 测试位置 256、1000、1500、
  2000 的 RMSE 分别约 2.385、3.530、3.606、2.695 LSB；原因尚未定位，不能声称只是舍入误差。
- 与旧应用独立保存的图片 0060 全 266 个位置对照：RMSE 0.826 LSB，93.93% 维度相同，
  第一个不同位置为绝对位置 239。不能把旧应用快照与本批逐位混用。

因此，**采集已完成，本批是固定 prefill 协议下可复现的监督数据**，可做该协议内的逆向模型实验。
若用于现有候选搜索路径，仍需校准执行路径差异；协议中的通用 `training_ready` 保守保留为 false，
不表示采集数据是模拟数据，而是尚未通过跨执行路径验收。此次没有运行模型训练。

审查文件：[validation.json](metadata/validation.json)、
[canonical_replay_control.json](metadata/canonical_replay_control.json)、
[replay_control.json](metadata/replay_control.json)、
[chat_capture_control.json](metadata/chat_capture_control.json)。
模型/运行时/APK 哈希见 `device/device.json`，采集源码快照与原始哈希匹配，保存在 `source/`。
8 项原有 backend 单元测试通过；原生库及测试 APK 构建通过。

早先逐 token 的中断采集保存在 `data/20261022_pilot`，不是本报告的完成数据集。
