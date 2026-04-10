import os
import cv2
import json
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from paddleocr import PaddleOCR

class DrawingProcessor:
    def __init__(self, model_path=None):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        self.cfg.MODEL.WEIGHTS = model_path if model_path else "model_final.pth"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.DEVICE = "cpu"
        
        try:
            self.predictor = DefaultPredictor(self.cfg)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.predictor = None
            
        self.ocr = None
        # Thứ tự nhãn đã được fix: 0-Note, 1-PartDrawing, 2-Table
        self.classes = {0: "Note", 1: "PartDrawing", 2: "Table"}

    def get_ocr_engine(self):
        if self.ocr is None:
            # Ở bản 2.7.0, các tham số này hoạt động cực kỳ ổn định
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        return self.ocr

    def process_image(self, im):
        if self.predictor is None or im is None:
            return {"error": "Model or Image error"}, None
            
        outputs = self.predictor(im)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        objects = []
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            cls_name = self.classes.get(int(classes[i]), "Unknown")
            
            x1, y1, x2, y2 = map(int, box)
            h, w = im.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            crop = im[y1:y2, x1:x2]
            
            ocr_content = ""
            if cls_name in ["Note", "Table"] and crop.size > 0:
                try:
                    ocr_engine = self.get_ocr_engine()
                    # Bản 2.7.0 hỗ trợ tham số cls=True rất tốt
                    ocr_res = ocr_engine.ocr(crop, cls=True) 
                    if ocr_res and ocr_res[0]:
                        lines = [line[1][0] for line in ocr_res[0]]
                        ocr_content = "\n".join(lines)
                except Exception as e:
                    ocr_content = f"[OCR Error: {str(e)[:100]}]"
            
            objects.append({
                "id": i + 1,
                "class": cls_name,
                "confidence": float(score),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "ocr_content": ocr_content
            })
            
        return {"objects": objects}, self.draw_viz(im, objects)

    def draw_viz(self, im, objects):
        v_im = im.copy()
        colors = {"PartDrawing": (255, 0, 0), "Note": (0, 255, 0), "Table": (0, 0, 255)}
        for obj in objects:
            b = obj["bbox"]
            color = colors.get(obj["class"], (255, 255, 255))
            cv2.rectangle(v_im, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 4)
            label = f"{obj['class']} {obj['confidence']:.2f}"
            cv2.putText(v_im, label, (b["x1"], b["y1"]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        return v_im
