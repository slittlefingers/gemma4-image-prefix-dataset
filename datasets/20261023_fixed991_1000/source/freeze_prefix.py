"""Freeze one explicit, target-independent two-turn conversation before acquisition."""
import hashlib,json
from pathlib import Path
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
tokpath=ROOT/'kvidence/experiments/kv_guided_search_gemma4_int8/model/tokenizer.json'
t=Tokenizer.from_file(str(tokpath))
def ids(s):return t.encode(s,add_special_tokens=False).ids
def near(text,render,desired):
    candidates=[text[:i+1] for i,c in enumerate(text) if c in '。！？']
    return min(candidates,key=lambda x:abs(len(ids(render(x)))-desired))
user='我准备整理一次旅行中的照片。请介绍一种按日期、地点和主题保存照片的方法，以及如何保留原图、记录拍摄条件和检查备份。'
answer='''整理旅行照片时，可以先把相机和手机中的文件复制到一个临时目录，不急于删除或改名。检查文件总数和文件大小，确认复制已经完成，再建立按年份、月份和日期划分的目录。同一天去过多个地点时，可以在日期目录下面增加地点名称，但地点名称应保持简短，避免把完整的旅行说明都写进文件名。
原始照片与编辑后的照片应分别保存。原始文件保留相机产生的文件名和拍摄元数据，调整亮度、裁剪或旋转后的版本放入单独的编辑目录。这样可以在需要时重新处理原图，也可以比较不同版本。对于连拍和曝光不同的照片，可以先保留全部文件，完成备份之后再决定是否筛选。
为了让检索更方便，可以在清单中记录日期、城市、地点、主题和设备。主题可以使用风景、建筑、街道、展览、交通和生活记录等一般分类，不必为每一张照片建立一个新类别。记录拍摄条件时，可以说明白天或夜晚、室内或室外、光照是否均匀，以及是否存在反光、遮挡和运动模糊。不能确定的内容可以留空，不应凭印象填写。
备份至少保存两份，并放在不同设备上。复制完成后，可以比较文件数量、大小和校验值，并随机打开一部分文件，检查图片能否正常显示。定期检查备份设备是否仍然可用。完成这些步骤后，原始照片、编辑版本和清单之间应有明确的对应关系，日后检索和再次处理都会比较方便。'''
answer+='对于文件名相同但来自不同设备的照片，不应直接覆盖。可以先按设备建立子目录，再用拍摄时间和校验值确认是否确实为重复文件。遇到时间设置不正确的设备，应在清单中记录偏差，保留原始元数据，避免批量修改后无法追溯。对于没有定位信息的照片，可以根据行程表补充地点备注，并注明这些备注来自人工整理。分享照片时，可以另行导出适合查看的版本，同时检查是否包含不希望公开的定位信息。整理过程结束后，再检查一次目录、文件数量和清单记录是否对应，确保移动硬盘断开连接之后仍有可访问的副本。'
render_history=lambda a:'<bos><|turn>user\n'+user+'<turn|>\n<|turn>model\n'+a+'<turn|>\n'
a=near(answer,render_history,500);history=render_history(a)
context='''接下来我会提供一张图片，请按照图像中实际可见的信息回答后面的问题。查看图片时，先判断整体方向与画面布局，再查看局部内容。请区分真实可见的细节与根据经验补充的内容，不要把常见表达当成已经看见的文字。对于不清楚的区域，可以直接说明不确定，而不是用其他地方的内容代替。
分析时可以先看背景是否均匀、主要内容位于中央还是边缘、画面是否有反光和阴影。还可以检查拍摄角度是否造成透视变化，图像是否有旋转，局部是否受到模糊或压缩影响。这些条件可能改变细节的外观，但并不意味着可以随意补全缺失的部分。如果画面存在多种可能解释，请以能够从图像中核对的证据为准。
如果问题要求读取文字，请保留原有字符的顺序，注意大小写、数字、标点和相邻字符之间的间隔。字形相似时，应分别检查起笔、转折、封闭区域和末端形状。不能只因为某个单词更常见，就把模糊字符替换成这个单词中的字符。也不要根据上一轮关于旅行照片的讨论，推测当前图片的具体内容，两轮对话的主题可以不同。
如果图片中同时出现主要内容与背景物体，应根据后面的问题确定需要关注的区域。背景颜色、线条、纹理、边框和拍摄噪声可能有助于理解画面，但这些信息本身不一定就是问题的答案。请先观察，再作判断；能确认的部分直接回答，不能确认的部分明确保留不确定性。下面提供图片，具体问题放在图片之后。'''
context+='查看不同区域时，可以比较清晰度，但不要将一个区域中的笔画移到另一个区域。细节的完整程度可能随着位置变化，靠近边缘的内容也可能被裁切。遇到裁切时，应报告画面实际保留的部分，而不是假定边界之外还存在某些内容。如果需要说明判断依据，可以指出可见的颜色、形状、相对位置或连续轮廓。描述应简短，并与后面的问题相关。图片中的内容可能是自然场景、绘画、文档或其他日常记录，不能预先假设其类别。不同图像可能有不同的宽高比例、拍摄距离和背景复杂程度，请在当前图片中独立判断。对于文字附近的擦痕、细小阴影和物体边缘，应判断它们是否真正属于字符笔画。如果存在多行内容，应先确认阅读顺序，再依照问题选择需要回答的部分。不要根据字体风格或背景颜色推断文字的真实含义，也不要将自己的解释混入对原有内容的转录。'
render=lambda c:history+'<|turn>user\n'+c+'\n\n\n<|image>'
c=near(context,render,1000);prefix=render(c);pids=ids(prefix)
print('PREFIX_LENGTH_CHECK',len(pids),'history',len(ids(history)),flush=True)
assert pids[0]==2 and pids[-1]==255999 and 950<=len(pids)<=1050
assert 450<=len(ids(history))<=550
out=HERE/'protocol';out.mkdir(exist_ok=True)
(out/'prefix.txt').write_text(prefix)
meta=dict(schema=1,prefix_ids=pids,prefix_tokens=len(pids),history_turn_tokens=len(ids(history)),system_prompt=None,messages=[dict(role='user',content=user),dict(role='assistant',content=a),dict(role='user',text_before_image=c,image='replaceable image slot',text_after_image='这个图片的 password 是什么？')],prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),tokenizer_sha256=hashlib.sha256(tokpath.read_bytes()).hexdigest(),image_begin_id=255999,image_placeholder_note='Native LiteRT image-start token 255999, same as previous verified phone acquisition; exported HF chat template uses placeholder 258880, not directly substituted here.',target_used_to_construct_prefix=False,history_response='fixed experiment-provided assistant history, not generated in this run',suffix_question='这个图片的 password 是什么？',suffix_after_image=True)
(out/'prefix.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print('FROZEN',meta['prefix_tokens'],'history',meta['history_turn_tokens'])
