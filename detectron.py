import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import _create_text_labels, GenericMask, VisImage


class Detectron():
    def __init__(self):
        self.cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, im, label):
        outputs = self.predictor(
            im)  # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        # print(outputs["instances"].pred_classes)
        # print(outputs["instances"].pred_boxes)
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        predictions = outputs["instances"].to("cpu")
        out = v.draw_instance_predictions(predictions)
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores,
                                     MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes", None))
        print(labels, classes)
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, im.shape[0], im.shape[1]) for x in masks]
        else:
            masks = None
        cv2.imshow("out", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

        mask = np.zeros(im.shape[:2])
        #TODO filter the mask keeping only the indecies of label
        # hint: you can access the masks by their index, like masks[idndex].mask 
        
        #to visualize the masked image
        mask = mask.astype(np.uint8)
        masked_image = cv2.bitwise_and(im, im, mask=mask)
        cv2.imshow("with mask applied", masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return mask


if __name__ == "__main__":
    im = cv2.imread("People.png")
    cv2.imshow("input", im)
    det = Detectron()
    det.predict(im, "person")
