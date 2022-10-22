import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_mask_rcnn(num_classes=2, pretrained=True):
    # Load a COCO-pretrained instance of Mask-RCNN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=pretrained)

    # Save the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pretrained predictor with a new instance that
    # generates the specified number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Save the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new instance that
    # generates masks pertaining to the number of classes specified
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
