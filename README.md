# Synthetic 3D Dataset Generator

A small tool for generating labelled 3D scenes of geometric objects. You run it, a window pops up with all the knobs, you set things up, hit Generate, and a few minutes later you have a folder full of PNGs and a CSV with the ground truth.

```
python syntheticdata.py
```

## What you get

Every scene is a handful of objects (cubes, spheres, cylinders, pyramids) sitting on a floor, rendered from a random camera angle. Each run produces:

- `scene_00000.png`, `scene_00001.png`, ... the rendered images
- `labels.csv` one row per image with shape counts and full per-object metadata

The CSV is straightforward:

```
filename,cube,sphere,cylinder,pyramid,objects
scene_00000.png,2,1,0,1,"[{"shape": "cube", "color": "red", ...}, ...]"
```

The four shape columns are counts. The `objects` column is a JSON list with every object's shape, colour, size category, and exact scale. To load it:

```python
import pandas as pd, json

df = pd.read_csv("synthetic_3d_dataset/labels.csv")
df["objects"] = df["objects"].apply(json.loads)
```

## The GUI

### Output

- **Directory** where the PNGs and CSV go. Type a path or hit Browse.
- **Scenes** how many images to generate.
- **Resolution** pixel size of each image. If you're training a ResNet-18, just set this to 224 and skip a resize later.

### Objects per scene (min, max)

For each shape you pick a min and max count. The actual count is drawn uniformly between them per scene. Min 0 means the shape can be absent entirely, min 1 guarantees it always shows up. If you want every shape in every scene, set them all to min 1.

### Layout and camera

- **Spread** how far from the centre objects can sit. Higher spreads them out, lower clusters them and gives you more occlusion.
- **Min distance** minimum gap between objects. Drop this if you want more overlap (overlap is good, it shows up in real images too).
- **Camera distance** how far the camera sits from the scene. 2.5 keeps everything in frame and you probably don't need to touch it.

### Camera angle

The horizontal angle is fully random every scene. You only control the vertical range:

- **Elevation min/max** drawn uniformly between these. The default 30 to 60 is a good range. Go below 20 and you start clipping the back of the scene into the floor.

### Augmentation

Everything is on by default. The whole point of synthetic data is to throw enough variation at the model that it doesn't learn shortcuts.

- **Lighting** randomises ambient, diffuse, and specular intensity. Stops the model latching onto fixed shading.
- **Background grey** the colour of the area outside the floor shifts between light grey and white.
- **Floor colour** the base floor colour shifts each scene.
- **Floor pattern** about half the scenes get a checkerboard, stripes, or noise texture on the floor with two random colours. The other half stay solid.
- **Scale jitter** each object's size gets multiplied by a random factor between 0.75 and 1.25. Keeps the model from using pixel size as a cue.
- **Full 3D rotation** objects rotate freely around all three axes instead of just yaw. A cube on its corner looks very different from one sitting flat.
- **Colour jitter** plus or minus 25 on each RGB channel of the object colour, so "red" isn't always the same red.
- **Camera zoom jitter** camera distance varies by 10% per scene.

## A reasonable starting point

```
Scenes:        5000
Resolution:    224 x 224
Objects:       min 1, max 2 for every shape
Spread:        4.0
Min distance:  1.5
Elevation:     30 to 60
All augmentations: on
```

## Loading it into PyTorch

```python
import pandas as pd, json
from PIL import Image
import torch
from torch.utils.data import Dataset

class ShapeDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.shape_names = ["cube", "sphere", "cylinder", "pyramid"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.img_dir}/{row['filename']}").convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(
            [int(row[s] > 0) for s in self.shape_names], dtype=torch.float
        )
        return img, label
```

Multi-label, so use `BCEWithLogitsLoss`.
