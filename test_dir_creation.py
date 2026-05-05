from pathlib import Path
import random
import shutil

train_img_dir = Path('./data/cityscapes/images/train')
train_lbl_dir = Path('./data/cityscapes/gtFine/train')
test_img_dir  = Path('./data/cityscapes/images/test')
test_lbl_dir  = Path('./data/cityscapes/gtFine/test')

SUFFIXES = [
    '_gtFine_color.png',
    '_gtFine_instanceIds.png',
    '_gtFine_labelIds.png',
    '_gtFine_labelTrainIds.png',
    '_gtFine_polygons.json',
]

# 🚫 SAFETY CHECK — stop if test already populated
existing_test = list(test_img_dir.rglob('*_leftImg8bit.png'))

if len(existing_test) > 0:
    print(f"⚠️ Test set already exists with {len(existing_test)} images. Skipping split.")
    print(f"Number of images in train set: {len(list(train_img_dir.rglob('*_leftImg8bit.png')))}")
    print(f"Number of images in test set: {len(existing_test)}")
    exit()

# ── collect train images ──
images = list(train_img_dir.rglob('*_leftImg8bit.png'))

random.seed(42)
random.shuffle(images)

test_size = int(len(images) * 0.1)
test_images = images[:test_size]

moved = 0

for img_path in test_images:
    city = img_path.parent.name
    base = img_path.name.replace('_leftImg8bit.png', '')

    (test_img_dir / city).mkdir(parents=True, exist_ok=True)
    (test_lbl_dir / city).mkdir(parents=True, exist_ok=True)

    # move image
    dest_img = test_img_dir / city / img_path.name
    shutil.move(img_path, dest_img)

    # move annotations
    for suffix in SUFFIXES:
        src = train_lbl_dir / city / f"{base}{suffix}"
        dest = test_lbl_dir / city / f"{base}{suffix}"

        if src.exists():
            shutil.move(src, dest)

    moved += 1

print(f"✅ Successfully moved {moved} images to test set")