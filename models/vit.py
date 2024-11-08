import torch.nn as nn
import timm

class ViT(nn.Module):
    """
    Initializes a Vision Transformer using timm.

    Args:
        model_name (str): The name of the timm ViT model.
        num_classes (int): The number of output classes.
        freeze (bool): Whether to freeze layers for training or not
    """
    def __init__(self, model_name: str, num_classes: int, freeze: bool):
        super(ViT, self).__init__()

        self.model = timm.create_model(model_name=model_name, pretrained=True)
        self.model.reset_classifier(num_classes)

        if freeze:
            for name, param in self.model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        else: 
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)