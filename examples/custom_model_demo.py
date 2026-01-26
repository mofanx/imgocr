# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Demo for loading custom model paths.
"""
import time
import sys

sys.path.append('..')
from imgocr import ImgOcr
from imgocr import draw_ocr_boxes

if __name__ == "__main__":
    # 示例1: 使用自定义检测模型和识别模型的绝对路径
    # 用户可以指定自己训练或下载的模型路径
    custom_det_model = "/path/to/your/custom_det_model.onnx"
    custom_rec_model = "/path/to/your/custom_rec_model.onnx"

    # 方式: 同时指定模型路径和字典文件路径
    # m = ImgOcr(
    #     det_model_path=custom_det_model,
    #     rec_model_path=custom_rec_model,
    # )

    # 实际运行示例: 使用默认模型（如果没有自定义模型可用）
    print("使用默认模型进行OCR...")
    m = ImgOcr(use_gpu=False)

    img_path = "data/11.jpg"
    s = time.time()
    result = m.ocr(img_path)
    e = time.time()
    print("total time: {:.4f} s".format(e - s))
    print("result:", result)
    for i in result:
        print(i['text'])

    # draw boxes
    draw_ocr_boxes(img_path, result, '11_custom_model_box.jpg')
    print('Save result to 11_custom_model_box.jpg')
