from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import config
from dataset import TrabecularMeshworkDataset
from model_forward import model_forward
from models.factory import create_model
from tqdm import tqdm


MODELS=[
    {"name":"monai_flexible_unet_b1",
     "checkpoint":config.OUTPUT_DIR/"monai_flexible_unet_b1_best.pt"},
    {"name":"monai_flexible_unet_b4",
     "checkpoint":config.OUTPUT_DIR/"monai_flexible_unet_b4_best.pt"},
    {"name":"monai_flexible_unet_b7",
     "checkpoint":config.OUTPUT_DIR/"monai_flexible_unet_b7_best.pt"}
]

SPLIT=config.TEST_SPLIT
OUTPUT_DIR=config.OUTPUT_DIR/"raw_predictions"/SPLIT
TTA_MODES=["none","hflip","vflip","hvflip"]

def load_model(spec,device):
    old=config.MODEL_NAME
    config.MODEL_NAME=spec["name"]
    model=create_model().to(device)
    ckpt=torch.load(spec["checkpoint"],map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    config.MODEL_NAME=old
    return model

def transform(x,mode):
    if mode=="hflip": return torch.flip(x,[3])
    if mode=="vflip": return torch.flip(x,[2])
    if mode=="hvflip": return torch.flip(x,[2,3])
    return x

def forward_tta(model,batch,device,mode):
    if mode=="none": return model_forward(model,batch,device)
    b=dict(batch)
    b["image"]=transform(batch["image"],mode)
    return transform(model_forward(model,b,device),mode)

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    ds=TrabecularMeshworkDataset(split=SPLIT,training=False)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=config.NUM_WORKERS)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets=OUTPUT_DIR/"targets"
    targets.mkdir(exist_ok=True)
    names=[]
    for mi,spec in enumerate(MODELS):
        model=load_model(spec,device)
        print(f"Collecting {spec['name']}")
        with torch.inference_mode():
            for batch in tqdm(loader):
                name=batch["name"][0]
                for tta in TTA_MODES:
                    d=OUTPUT_DIR/spec["name"]/tta
                    d.mkdir(parents=True,exist_ok=True)
                    logits=forward_tta(model,batch,device,tta)[0].float().cpu().numpy()
                    np.save(d/f"{name}.npy",logits.astype(np.float16))
                if mi==0:
                    np.save(targets/f"{name}.npy",batch["target"][0].numpy().astype(np.int16))
                    names.append(name)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (OUTPUT_DIR/"names.txt").write_text("\n".join(names))
    print(f"Saved {len(names)} samples for {len(MODELS)} models to {OUTPUT_DIR}")


if __name__=="__main__":
    main()
