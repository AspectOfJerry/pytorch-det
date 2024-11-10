import torch
import torchvision


def new_model(out_features):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features=in_features, out_features=out_features * 4, bias=True)
    return model


def new_optimizer(model, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer


def new_scheduler(optimizer, step_size, gamma):
    return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
