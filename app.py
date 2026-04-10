import gradio as gr
import cv2
import os
import json
import numpy as np
from inference_utils import DrawingProcessor

processor = DrawingProcessor(model_path="model_final.pth")

def detect_and_ocr(image):
    if image is None: 
        return None, "Vui lòng tải ảnh lên", "N/A"
    
    # Chuyển từ RGB sang BGR trực tiếp
    im_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Xử lý ảnh (không cần qua file tạm)
    result_json, viz_img = processor.process_image(im_bgr)
    
    ocr_display = ""
    for obj in result_json.get("objects", []):
        if obj.get("ocr_content"):
            ocr_display += f"=== {obj['class']} (ID: {obj['id']}) ===\n{obj['ocr_content']}\n\n"
            
    if not ocr_display:
        ocr_display = "Không có nội dung OCR."
        
    return cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB), json.dumps(result_json, indent=4), ocr_display

with gr.Blocks() as demo:
    gr.HTML("<div style='text-align: center;'><h1>📐 Engineering Drawing Detection & OCR</h1></div>")
    with gr.Row():
        input_img = gr.Image(label="Upload bản vẽ", type="numpy")
        output_img = gr.Image(label="Kết quả Visualization")
    btn = gr.Button("🚀 PHÁT HIỆN & TRÍCH XUẤT", variant="primary")
    with gr.Row():
        json_output = gr.Code(label="JSON Data", language="json", lines=10)
        ocr_output = gr.Textbox(label="Nội dung OCR", lines=10)
            
    btn.click(fn=detect_and_ocr, inputs=input_img, outputs=[output_img, json_output, ocr_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
