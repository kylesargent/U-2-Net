import json
import numpy as np
from PIL import Image, ImageDraw
import os
from tqdm import tqdm
import pickle

# polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
# width = ?
# height = ?

metadata = []
for f in tqdm(sorted(os.listdir('data'))):
    if f.endswith('.json') and f != 'split.json':
        metadata_this = {
            'filename': f.replace('.json', '.jpg'),
            'mask_filename': f.replace('.json', '_mask.png'),
        }
        masks = []
        polygons = []
        with open('data/'+ f) as json_file:
            data = json.load(json_file)
            # assert len(data['shapes']) == 1, f'{f} has {len(data["shapes"])} mask'
            for s in data['shapes']:
                width, height = data['imageWidth'], data['imageHeight']
                mask_img = Image.new('L', (width, height), 0)
                polygon = s['points']
                polygon = [tuple(x) for x in polygon]
                polygons.append(polygon)
                ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
                mask = np.array(mask_img)
                masks.append(mask)
            
            masks_all = np.stack(masks).max(axis=0)
            metadata_this['masks'] = masks
            metadata_this['polygons'] = polygons
        Image.fromarray(masks_all.astype(np.uint8) * 255).save('data/'+ f[:-4] + '_mask.png')
        metadata.append(metadata_this)

with open('data/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

np.random.seed(0)
idxs = np.arange(len(metadata))
# 167 examples
np.random.shuffle(idxs)
train_idxs = idxs[:132]
val_idxs = idxs[132:132 + 15]
test_idxs = idxs[132 + 15:]

# 399 examples
np.random.shuffle(idxs)
train_idxs = idxs[:350]
val_idxs = idxs[350: 350 + 25]
test_idxs = idxs[375:]

print(f'Train: {len(train_idxs)} Val: {len(val_idxs)} Test: {len(test_idxs)}')
with open('data/split.json', 'w') as f:
    d = {
        'train': [metadata[i]['filename'] for i in train_idxs],
        'val': [metadata[i]['filename'] for i in val_idxs],
        'test': [metadata[i]['filename'] for i in test_idxs],
    }
    json.dump(d, f, indent=4)

