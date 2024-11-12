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


def new_optimizer(model, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad
    )
    return optimizer


def new_scheduler(optimizer, step_size, gamma=0.1, last_epoch=-1):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=step_size,
        gamma=gamma,
        last_epoch=last_epoch
    )
    return scheduler
