"""Write the requested ten-example reconstruction report."""
import json

def make_report(rows,output):
    assert len(rows)==10 and [r['number'] for r in rows]==list(range(1,100,10))
    text=['# 测试图片还原：10 张样例','',
          '**从 100 张留出测试图中，按 ID 排序后每隔 10 张取一张：第 1、11、21……91 张。** 不根据重建质量挑选。使用原先冻结的第 11 轮模型，不重新训练或选择 checkpoint。','',
          '[实验总报告](../README.md) · [这 10 张的指标与来源](gallery_metrics.json) · [导出校验](export_audit.json)','',
          '整体上仍主要恢复模糊的明暗和大块布局，细节与文字明显丢失。完整 100 张测试集的平均 PSNR 为 13.28 dB、SSIM 为 0.412；这些是完整测试集指标，不是下方 10 张的平均，也不是图片恢复成功率。','',
          '每组左起：**参考图 / 联合 Transformer / 平滑后的分段线性链路 / 分段线性链路 / 真实 embedding 解码参考**。最后一列需要真实 embedding，仅供参考，不属于 K/V-only 恢复。','',
          '参考图是 native STB 图像宽高各减半后的像素，重建图与它尺寸相同。可点击单独 PNG 查看输出原尺寸。指标在保存为 8-bit PNG 之前计算。','',
          '| 测试编号 | 图片 ID | 联合模型 PSNR | SSIM | 对照 |','|---:|---|---:|---:|---|']
    for r in rows:
        m=r['metrics']['joint'];text.append(f"| {r['number']:03d} | {r['source_image_id']} | {m['psnr_db']:.3f} dB | {m['ssim']:.4f} | [查看](comparisons/{r['stem']}.png) |")
    for i,r in enumerate(rows,1):
        stem=r['stem'];text += ['',f"## 样例 {i} · 测试编号 {r['number']:03d}",'',
            f"图片 ID：`{r['source_image_id']}`；输出 {r['width']}×{r['height']}，{r['positions']} 个图像位置。",'',
            f"![样例 {i} 重建对照](comparisons/{stem}.png)",'',
            f"[参考图 PNG](images/{stem}_reference.png) · [联合模型重建 PNG](images/{stem}_joint.png)",'',
            '| 方法 | 像素 MSE ↓ | PSNR ↑ | SSIM ↑ |','|---|---:|---:|---:|']
        for k,label in [('joint','联合模型'),('smooth_separate_chain','平滑分段链路'),('separate_chain','分段链路'),('true_e','真实 embedding 参考')]:
            v=r['metrics'][k];text.append(f"| {label} | {v['mse']:.6f} | {v['psnr_db']:.3f} dB | {v['ssim']:.4f} |")
    text+=['','## 来源与复现','',
           '来源为 HierText 官方 train split，在本实验的 1000 张图片中按图片 ID 留出 100 张作为测试集；本报告从中固定间隔取 10 张。白板是额外的独立测试对象，不在这 10 张中。','',
           '原图 ID、原图和采集数组哈希、导出 PNG 哈希保存在 `gallery_metrics.json`。10 张图片的指标与此前发布的同图指标完全一致，最大差为 0。','',
           '图片与重建衍生图来自 [HierText](https://github.com/google-research-datasets/hiertext)，按 [CC BY-SA 4.0](../HIERText_LICENSE.txt) 发布。变换包括 native STB 缩放、半分辨率取样、模型重建和平滑处理。','',
           '已有匹配的冻结 checkpoint 后，在仓库根目录运行：','', '```bash',
           'python experiments/20261023_fixed991_rasr/export_test_gallery.py --checkpoints /path/to/original/results','```','',
           '脚本验证 checkpoint 哈希，然后导出同样的 10 张。模型 checkpoint 不包含在本次 Git 提交中。','']
    (output/'README.md').write_text('\n'.join(text),encoding='utf-8')
