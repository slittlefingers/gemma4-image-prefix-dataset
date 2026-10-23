固定前缀真实 NPU 数据集：1000 张训练来源图片与一个独立重点测试对象。

- 图片来自 HierText 官方训练集，共 1000 张，其中 306 张包含符合筛选条件的手写文字。此 Release 不包含原计划的其他 9000 张图片。
- 图片前固定 991 tokens，上一轮完整 user/assistant 对话为 511 tokens。前缀是构造的实验对话，完整文本和 token IDs 随包提供；不是提取的实际受害者会话。
- Gemma 4 E2B Google Tensor G5，真实手机 CPU 视觉编码器/adapter，真实手机 NPU 第 5 层（索引 4）原始 INT8 K/V。量化参数随包提供。
- 每张图保存整张空间网格、K/V、768 维 adapter 前 FP32 特征、1536 维 adapter 后 FP32 embedding，以及 native STB 预处理后的 RGB 像素标签。
- 固定前缀快照重放与重新计算完整前缀在控制图片上逐位一致，RMSE=0 LSB。这个结论不覆盖其他分块或候选查询路径。
- 图片后的统一问题为“这个图片的 password 是什么？”。训练采集止于图片 K/V，后缀记录在元数据中；重点测试图片实际执行了 EOI 和后续问题。

重点测试对象：用户明确授权公开的手机白板手写照片。原图、同一固定前缀下的观测 K/V、以及独立的真实像素/embedding 对照放在 `important_test/whiteboard/`，不出现在训练 index.json 中。不得用目标原图、真实 embedding 或文字来训练或选择 checkpoint。目标文字尚未独立核对标注，不把模型猜测作为文字真值。

1000 张采集图片均来自官方 train split；本批没有另外采集验证/测试图片。需要调参时，应从这 1000 张中按原始图片 ID 留出验证集，重点白板图片始终独立。尚未训练恢复网络，未报告目标恢复成功。

下载两个训练分包、元数据/重点测试对象包和 SHA256SUMS：

```bash
gh release download v2026.10.23-fixed-prefix-1000 --repo slittlefingers/gemma4-image-prefix-dataset
shasum -a 256 -c SHA256SUMS
for f in gemma4-fixed991-1000-*.tar.gz; do tar -xzf "$f"; done
```

三个包都解压到 `data/20261023_fixed991_1000/`。训练数组位于 samples/；重点测试对象位于 important_test/whiteboard/。无需把重点测试目录传入训练加载器。

训练图片衍生数据遵循 HierText 官方 CC BY-SA 4.0 来源说明；附官方 README 和 LICENSE，不把模型权重或运行时随包分发。用户提供的重点测试照片经授权公开，不自动适用 HierText 的数据许可证。
