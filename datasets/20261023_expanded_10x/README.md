# Gemma 4 image-prefix dataset: tenfold expansion

2026-10-23 真实 Pixel 10 / Gemma 4 E2B NPU 图片前缀数据扩充版。

| Split | Sequences | Unique images | Target positions |
|---|---:|---:|---:|
| Train | 60 | 480 | 118,620 |
| Validation | 10 | 80 | 19,770 |
| Test | 10 | 80 | 19,770 |
| Total | 80 | 640 | 158,160 |

位置样本恰好为先导版本的 10 倍，独立图片从 56 张增加到 640 张。源图片池含 1000 张 COCO val2017 图片，实际采集使用其中 640 张。保留原有图片的 split 归属；新图片按图片划分，三个集合的源文件 SHA256 无重叠。

- 前缀长度 10–2000；第 5 层原始 INT8 K/V，512+512 维及量化参数。
- adapter 前 768 维、adapter 后 1536 维特征均为 float32。
- 手机 CPU 视觉编码器和 adapter、真实手机 NPU 语言解码器；保持 causal prefill_128 采集协议。
- 无自然语言问题或 system prompt，沿用构造的多图片 user-turn 模板。
- 文件哈希、维度、位置、标签对应关系和图片划分校验通过。新 NPU 会话保持原分块重放 train_000 的 1982 个图片位置，K/V 逐位一致，RMSE=0 LSB。
- 本版尚未重新训练模型；旧版训练与重建结果不能当作扩充版结果。

下载两个数据包和 SHA256SUMS，然后执行：

```bash
shasum -a 256 -c SHA256SUMS
tar -xzf gemma4-image-prefix-20261023-expanded-core.tar.gz
tar -xzf gemma4-image-prefix-20261023-expanded-images.tar.gz
```

两个包均解压至 `data/20261023_expanded_10x/`。core 包含序列数组、元数据、校验报告、采集日志和源码快照；images 包含每张图片的特征数组及来源/许可元数据。原始 JPEG、Gemma 权重和运行时不随包分发。SHA256SUMS 校验两个数据包。

限制：仅使用真实历史前缀；尚未验证预测前缀下的恢复质量。构造的多图片上下文不代表自然对话。新版本仅对保持原分块的执行路径进行了重放检查，不能据此推断改变分块或候选查询路径也逐位一致。图片及衍生特征的使用应遵循各自来源条款。
