```
# %%
import sys
import os
sys.path.append("models")

import csv
from sklearn import metrics, cluster
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import math
import PIL
from PIL import Image
import argparse
import pickle


from torchvision import transforms
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler
import patchcore.datasets.mvtec as mvtec
from patchcore.utils import Matrix_Alpha_Unsupervised, Matrix_Alpha_Supervised
from patchcore.patchcore import AnomalyClusteringCore  # This is originated from PatchCore and it is modified a little bit.
import test


from munkres import Munkres
LOGGER = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
backbone_names = ["dino_vitbase8"]
layers_to_extract_from = ["blocks.10", "blocks.11"]

# backbone_names = ["wideresnet50"]
# layers_to_extract_from = ["layer2", "layer3"]

path = 'data/mvtec_ad'
pretrain_embed_dimension = 2048
target_embed_dimension = 4096
output_dir = "outputs"
patchsize = 3
tau = 2
train_ratio = 1
supervised = 'unsupervised'
dataset = 'mvtec_ad'
category = "bottle"

# %%
save_path = output_dir + "/" + dataset + "/" + backbone_names[0] + "/" + supervised
print("{:-^80}".format(category + ' start ' + supervised))
# 参数初始化
faiss_on_gpu = True
faiss_num_workers = 4
input_shape = (3, 224, 224)
anomaly_scorer_num_nn = 5
sampler = patchcore.sampler.IdentitySampler()
backbone_seed = None
backbone_name = backbone_names[0]

loaded_patchcores = []

# 加载数据集，dataloader
test_dataset = mvtec.MVTecDataset(source=path, split=mvtec.DatasetSplit.TEST,
                                    classname=category, resize=256, imagesize=224)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


if len(backbone_names) > 1:
    layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
    for layer in layers_to_extract_from:
        idx = int(layer.split(".")[0])
        layer = ".".join(layer.split(".")[1:])
        layers_to_extract_from_coll[idx].append(layer)
else:
    layers_to_extract_from_coll = [layers_to_extract_from]
layers_to_extract_from = layers_to_extract_from_coll[0]

if ".seed-" in backbone_name:
    backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
        backbone_name.split("-")[-1]
    )
backbone = patchcore.backbones.load(backbone_name)
backbone.name, backbone.seed = backbone_name, backbone_seed

nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

# 实例化模型对象
anomalyclusteringcore_instance = AnomalyClusteringCore(device)
anomalyclusteringcore_instance.load(
    backbone=backbone,
    layers_to_extract_from=layers_to_extract_from,
    device=device,
    input_shape=input_shape,
    pretrain_embed_dimension=pretrain_embed_dimension,
    target_embed_dimension=target_embed_dimension,
    patchsize=patchsize,
    featuresampler=sampler,
    anomaly_scorer_num_nn=anomaly_scorer_num_nn,
    nn_method=nn_method,
)


anomalies = test_dataset.anomaly_types
anomalies.remove("good")
anomalies.insert(0,"good")
np.save(f"./embedding/{category}/anomalies", anomalies)

anomalies = {key: index for index, key in enumerate(anomalies)}

features = []
masks = []
for data in test_dataloader:
    image = data["image"].to(device)
    mask = data["mask"]
    feature = anomalyclusteringcore_instance._embed(image, supervised)
    features.append(feature)
    mask = torch.nn.MaxPool2d(kernel_size=8)(mask).flatten()
    mask = torch.where(mask>0, anomalies[data["anomaly"][0]], 0)
    masks.append(mask)
features = torch.tensor(features).reshape(-1,4096).numpy()
masks = torch.concat(masks).numpy()

os.makedirs(f"./embedding/{category}", exist_ok=True)
np.save(f"./embedding/{category}/mask", masks)
np.save(f"./embedding/{category}/{backbone_name}_emb", features)
```

```
# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle

backbone_name = "dino_vitbase8"
# backbone_name = "wideresnet50"
category = "bottle"

embedding = np.load(f"./embedding/{category}/{backbone_name}_emb.npy")
anomlabel = np.load(f"./embedding/{category}/mask.npy")
anom = list(np.load(f"./embedding/{category}/anomalies.npy"))
print(anom)

vis_anom = [3,2,1]

centers = []
for i in vis_anom:
    mean = np.mean(embedding[anomlabel==i], axis=0)
    centers.append(mean)
centers = np.stack(centers, axis=0)

# %%
def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - sum(np.dot(v, b)*b for b in basis)
        if (w > 1e-10).any():  
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)

vectors = [centers[1]-centers[0], centers[2]-centers[0]]
basis = gram_schmidt(vectors)

projected = embedding[:, np.newaxis, :] * basis[np.newaxis, :, :]
projected = np.sum(projected, axis=2)

# %%
for i in [0] + vis_anom:
    plt.scatter(
        projected[anomlabel==i,0],
        projected[anomlabel==i,1],
        label=anom[i],
        alpha=.2
        )

plt.legend()
plt.show()
# %%

```

```
self.anomaly_types = anomaly_types
```
