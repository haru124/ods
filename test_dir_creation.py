from pathlib import Path
import random
import shutil

# ── Paths ──
train_img_dir = Path('./data/cityscapes/images/train')
train_lbl_dir = Path('./data/cityscapes/gtFine/train')
test_img_dir  = Path('./data/cityscapes/images/test')
test_lbl_dir  = Path('./data/cityscapes/gtFine/test')

test_img_dir.mkdir(parents=True, exist_ok=True)
test_lbl_dir.mkdir(parents=True, exist_ok=True)

# ── Collect all training images ──
images = list(train_img_dir.rglob('*_leftImg8bit.png'))
print(f"Total train images: {len(images)}")

random.seed(42)
random.shuffle(images)

test_size   = int(len(images) * 0.1)
test_images = images[:test_size]
print(f"Test set size: {len(test_images)} images")

# ── Copy with skip-if-exists ──
copied   = 0
skipped  = 0
missing  = 0

SUFFIXES = [
    '_gtFine_color.png',
    '_gtFine_instanceIds.png',
    '_gtFine_labelIds.png',
    '_gtFine_labelTrainIds.png',
    '_gtFine_polygons.json',
]

for img_path in test_images:
    city      = img_path.parent.name
    base_name = img_path.name.replace('_leftImg8bit.png', '')

    # Create city subfolders inside test/
    (test_img_dir / city).mkdir(parents=True, exist_ok=True)
    (test_lbl_dir / city).mkdir(parents=True, exist_ok=True)

    # ── Copy image — skip if already exists ──
    dest_img = test_img_dir / city / img_path.name
    if dest_img.exists():
        skipped += 1
    else:
        shutil.copy(img_path, dest_img)
        copied += 1

    # ── Copy all 5 annotation files — skip if already exists ──
    for suffix in SUFFIXES:
        src  = train_lbl_dir / city / f"{base_name}{suffix}"
        dest = test_lbl_dir  / city / f"{base_name}{suffix}"
        if dest.exists():
            pass   # already there, skip
        elif src.exists():
            shutil.copy(src, dest)
        else:
            print(f"  Missing annotation: {src}")
            missing += 1

print(f"\nImages copied  : {copied}")
print(f"Images skipped : {skipped}  (already existed)")
print(f"Missing files  : {missing}")

# ── Correct verification — no extra 'test' in path ──
# Files are at:  images/test/cityname/filename.png
# NOT at:        images/test/test/cityname/filename.png
test_imgs  = list(test_img_dir.rglob('*_leftImg8bit.png'))
test_masks = list(test_lbl_dir.rglob('*_gtFine_labelTrainIds.png'))
test_jsons = list(test_lbl_dir.rglob('*_gtFine_polygons.json'))

print(f"\nVerification:")
print(f"  Images : {len(test_imgs)}")
print(f"  Masks  : {len(test_masks)}")
print(f"  JSONs  : {len(test_jsons)}")

if len(test_imgs) == len(test_masks) == len(test_jsons) == test_size:
    print(f"\n✅ All {test_size} samples complete")
elif len(test_imgs) > 0:
    print(f"\n⚠️  Counts don't match — some annotations may be missing")
else:
    print(f"\n❌ No files found — check paths")

if test_imgs:
    print(f"\nExample image path : {test_imgs[0]}")
    print(f"Example mask path  : {test_masks[0] if test_masks else 'NOT FOUND'}")