import faulthandler
faulthandler.enable()

import torch
from models import create_WSI_model
from dataset.dataset_survival_egmde import DataGeneratorTCGASurvivalWSIEGMDE
from torch.utils.data import DataLoader

# ----- Step 0: config setup -----
import yaml
from munch import Munch
cfg_path = 'configs/luad_sgcmde.yaml'
with open(cfg_path, 'r') as f:
    cfg = Munch.fromDict(yaml.safe_load(f))

print("✅ Config loaded")

# ----- Step 1: Model creation -----
model = create_WSI_model(cfg)
print("✅ Model created")
model.to(cfg.device)
print("✅ Model moved to device")

# ----- Step 2: Dataset creation -----
from os.path import join as pjoin
root = cfg.datasets.root_dir
fold = cfg.datasets.fold[0]  # Test only one fold

anno_path = pjoin(root, cfg.datasets.wsi_file_path)
clinical_path = pjoin(root, cfg.datasets.clinical_file_path)
train_ids_path = pjoin(root, cfg.datasets.folds_path, f"fold{fold}", "train.txt")

ds = DataGeneratorTCGASurvivalWSIEGMDE(
    anno_file_path=anno_path,
    wsi_id_path=train_ids_path,
    clinical_path=clinical_path,
    shuffle=False,
    with_coords=getattr(cfg.datasets, 'with_coords', False),
    cluster_label_path=None
)
print("✅ Dataset loaded:", len(ds))

# ----- Step 3: Try sample -----
sample = ds[0]
print("✅ One sample loaded")

# ----- Step 4: Dataloader test -----
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

for step, batch in enumerate(loader):
    print(f"✅ Batch {step} loaded")
    if step == 0:
        break

print("🎉 All tests passed")
