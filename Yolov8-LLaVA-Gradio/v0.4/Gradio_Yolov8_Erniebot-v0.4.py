from __future__ import annotations
import gradio as gr
import erniebot
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
history=[]
catorgory_now=gr.State(value=[history,'无'])

# 定义一个字典，将英文映射为中文
category_translations = {
    'bio': '生物',
    'plastic': '塑料',
    'rov': '潜水器',
    'timestamp': '时间戳',
    'unknown': '未知',
    'metal': '金属',
    'wood': '木头',
    'rubber': '橡胶',
    'cloth': '布料',
    'fishing': '渔具',
    'paper': '纸张'
}
class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.blue,
        secondary_hue: colors.Color | str = colors.emerald,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="repeating-linear-gradient(45deg, *primary_100, *primary_100 20px, *primary_400 20px, *primary_200 40px)",
            body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 20px, *primary_900 20px, *primary_900 40px)",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )


seafoam = Seafoam()

model = YOLO("best.pt")
def catorgorys(list,catorgory_now):
    catorgory_t=''
    ans='一共识别到了'+str(len(list))+'个物体，种类有：'
    for item in list:
        if category_translations[model.names[int(item)]] not in ans:
            ans+=category_translations[model.names[int(item)]]+','
            if category_translations[model.names[int(item)]]=='时间戳':
                continue
            catorgory_t+=category_translations[model.names[int(item)]]+'和'
    catorgory_now=gr.State(value=[history,catorgory_t[:-1]])
    
    return ans[:-1],catorgory_now
def predict_image(img, conf_threshold, iou_threshold,catorgory_now):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
    )
    if results:
        img_result = results[0].plot()
        img_result = cv2.cvtColor(np.array(img_result), cv2.COLOR_BGR2RGB)
        img_result = Image.fromarray(img_result)
        ans,catorgory_now=catorgorys(results[0].boxes.cls.tolist(),catorgory_now)
        return img_result,ans,catorgory_now
    else:
        return None,None,catorgory_now
demo1 = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
        gr.Slider(minimum=0, maximum=1, step=0.05, value=0.25, label="自信度阈值（低至高）"),
        gr.Slider(minimum=0, maximum=1, step=0.05, value=0.45, label="重叠度阈值（低至高）"),
        catorgory_now,
    ],
    outputs=[gr.Image(type="pil", label="识别结果"),gr.components.Textbox(label="识别物体种类", lines=3),catorgory_now],
    title="基于YOLOv8的海洋生物与垃圾智能识别系统",
    description="""
    <h3>欢迎使用我们的智能识别系统！</h3>
    <p>请上传一张海洋环境中的图片，系统将自动识别其中的生物或垃圾。</p>
    <p>通过调整下方的阈值，可以控制识别的精度与范围。</p>
    """,
    
)

def response(message,catorgory_now):
    erniebot.api_type = 'aistudio'
    erniebot.access_token = '557b19fc169d0603539aaccfacfc46f180d6f725'
    model = 'ernie-4.0-turbo-8k'
    messages=[{'role': 'user', 'content': '现在你接收到了一张含有'.join(catorgory_now.value[1])+'的照片，请你回答我的问题'},{'role': 'assistant', 'content': '好的，请给我一个具体的问题'}]
    history=catorgory_now.value[0]
    if history:
        for item in history:
            messages.append({'role': 'user', 'content': item[0]})
            messages.append({'role': 'assistant', 'content': item[1]})
    # 添加用户的新消息
    messages.append({'role': 'user', 'content': message})
    def main(model, messages):
        response = erniebot.ChatCompletion.create(
            model = model,
            messages = messages,
            top_p = 0.95,
            system='现在你是一个海洋智能助手，你要为用户介绍海洋环境，同时提出海洋保护、环境治理的建议，你说话时要用严谨、机械化的口吻',) # top_p影响生成文本的多样性，取值越大，生成文本的多样性越强,取值范围为[0, 1.0]
        result = response.get_result()
        return result,response
    answer, response = main(model, messages)
    history=messages[:].append({'role': 'assistant', 'content': 'answer'})
    catorgory_now=gr.State(value=[history,catorgory_now.value[1]])
    return answer,catorgory_now

# def response(message,catorgory_now):
#     return catorgory_now.value[1],catorgory_now
demo2 = gr.Interface(
    fn=response,
    inputs=[
        gr.Textbox(placeholder="请在这里提出你对这些生物的疑问", container=False, scale=7),
        catorgory_now  # 合并的state输入
    ],
    outputs=[
        gr.Textbox(lines=5,label="回复", container=False, scale=7),  # 消息回复
        catorgory_now                # 合并的state输出
    ],
    title="海洋生态介绍小助手",
)

def welcome():
    image_path = "Cover.jpg"  # 替换为你的图片路径

    return image_path
demo0 = gr.Interface(
    fn=welcome,
    inputs=None,
    outputs=gr.Image(type="filepath"),
    title="基于YOLOv8的海洋生物与垃圾智能识别系统",
    description="""
    <h1>欢迎使用我们的智能识别系统！</h1>
    """,
    live=True
)
with gr.Blocks(theme=seafoam) as demo:
    with gr.Tabs():
        with gr.TabItem("主页面"):
            demo0.render()
        with gr.TabItem("智能识别系统"):
            demo1.render()
        with gr.TabItem("介绍小助手"):
            demo2.render()
if __name__ == "__main__":
    demo.launch()