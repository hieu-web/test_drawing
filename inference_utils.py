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

    def calculate_iou(self, boxA, boxB):
        # box format: x1, y1, x2, y2
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def clean_ocr_text(self, text):
        if not text:
            return ""
        # Loại bỏ các ký tự rác phổ biến ở đầu/cuối
        text = text.strip()
        # Chuyển đổi các lỗi phổ biến (VD: O -> 0 trong chuỗi số)
        # Đây là ví dụ đơn giản, có thể mở rộng thêm regex chuyên sâu
        return text

    def postprocess_objects(self, objects, iou_threshold=0.5, containment_threshold=0.8):
        if not objects:
            return []
            
        # 1. Sắp xếp theo diện tích (Area) từ lớn đến nhỏ để ưu tiên các khung bao quát
        for obj in objects:
            b = obj["bbox"]
            obj["area"] = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            
        objects = sorted(objects, key=lambda x: x["area"], reverse=True)
        keep = []
        
        while len(objects) > 0:
            current = objects.pop(0)
            is_redundant = False
            
            for saved in keep:
                boxA = [current["bbox"]["x1"], current["bbox"]["y1"], current["bbox"]["x2"], current["bbox"]["y2"]]
                boxB = [saved["bbox"]["x1"], saved["bbox"]["y1"], saved["bbox"]["x2"], saved["bbox"]["y2"]]
                
                iou = self.calculate_iou(boxA, boxB)
                
                # Tính độ chứa đựng (Containment): current nằm trong saved bao nhiêu %
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                containment = interArea / float(current["area"]) if current["area"] > 0 else 0

                # Điều kiện loại bỏ:
                # 1. Trùng lặp IOU cao (cùng class)
                if current["class"] == saved["class"] and iou > iou_threshold:
                    is_redundant = True
                    break
                
                # 2. Box hiện tại nằm lọt thỏm trong một box khác đã lưu (Containment cao)
                if containment > containment_threshold:
                    is_redundant = True
                    break
                    
            if not is_redundant:
                keep.append(current)
            
        # 2. Làm sạch văn bản OCR
        for obj in keep:
            obj["ocr_content"] = self.clean_ocr_text(obj["ocr_content"])
            
        # 3. Đánh lại ID
        for i, obj in enumerate(keep):
            obj["id"] = i + 1
            
        return keep

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
            
        # Áp dụng Post-processing
        objects = self.postprocess_objects(objects)
            
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
