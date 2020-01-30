
import torch
from torch import nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import requests
import io
from core.util import logging
from time import time


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. \
    `full` for rank-1 tensor"

    def __init__(self, full: bool = False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        base_model = models.resnet18(pretrained=False)
        base_model = nn.Sequential(*[m for m in base_model.children()][:-2])
        head = nn.Sequential(
         AdaptiveConcatPool2d(),
         Flatten(),
         nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True),
         nn.Dropout(p=0.25),
         nn.Linear(in_features=1024, out_features=512, bias=True),
         nn.ReLU(inplace=True),
         nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
         nn.Dropout(p=0.5),
         nn.Linear(in_features=512, out_features=2, bias=True)
         )
        self.model = nn.Sequential(base_model, head)

    def forward(self, x):
        x = self.model(x)
        return x

    def load_state_dict(self, *args, **kwargs):
        logging.info(f"Loading State Dict of Model")
        st = time()
        self.model.load_state_dict(torch.load(*args, **kwargs))
        logging.info(f"Model loaded into memory in {round(time() - st, 2)} secs")

    def set_eval(self):
        logging.info("Set Model to Eval Model")
        self.model = self.model.eval()

    def to_cpu(self):
        self.model = self.model.cpu()

    def predict(self, x):
        x = x.unsqueeze(0)
        confidences = self.model(*[x])
        confidences = torch.softmax(confidences, dim=1).detach().numpy()[0]
        return confidences

    def preprocess_image(self, im):
        im = im.convert('RGB')
        im = transforms.Resize(256)(im)
        im = transforms.CenterCrop(224)(im)
        im = transforms.ToTensor()(im)
        im = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])(im)
        return im


class CatsDogsModel(Model):

    def __init__(self, model_path):
        super(CatsDogsModel, self).__init__()
        self.load_state_dict(f=model_path,
                            map_location="cpu")
        self.set_eval()

    def predict(self, x):
        confidences = super().predict(x)
        pred = np.argmax(confidences)

        res = {
            "cat_or_dog": "cat" if pred == 0 else "dog",
            "confidence": str(round(confidences[pred], 2))
        }
        return res

    def score(self, im):
        im = self.preprocess_image(im)
        prediction = self.predict(im)
        return prediction

    def score_image_from_path(self, img_path):
        im = Image.open(img_path)
        prediction = self.score(im)
        return prediction

    def score_image_from_url(self, url):
        img_request = requests.get(url, stream=True)
        im = Image.open(io.BytesIO(img_request.content))
        prediction = self.score(im)
        return prediction
