from pathlib import Path
import random
import shutil

# Paths
train_img_dir = Path('./data/cityscapes/images/train')
train_lbl_dir = Path('./data/cityscapes/gtFine/train')

test_img_dir = Path('./data/cityscapes/images/test')
test_lbl_dir = Path('./data/cityscapes/gtFine/test')

# Create base test dirs
test_img_dir.mkdir(parents=True, exist_ok=True)
test_lbl_dir.mkdir(parents=True, exist_ok=True)

# Collect all images recursively
images = list(train_img_dir.rglob('*_leftImg8bit.png'))

print(f"Total train images: {len(images)}")

# Shuffle (reproducible)
random.seed(42)
random.shuffle(images)

# Split
split_ratio = 0.1
test_size = int(len(images) * split_ratio)

test_images = images[:test_size]

print(f"Creating test set with {len(test_images)} images")

# Process each image
for img_path in test_images:
    city = img_path.parent.name
    base_name = img_path.name.replace('_leftImg8bit.png', '')

    # Create city folders
    (test_img_dir / city).mkdir(parents=True, exist_ok=True)
    (test_lbl_dir / city).mkdir(parents=True, exist_ok=True)

    # Copy image
    shutil.copy(
        img_path,
        test_img_dir / city / img_path.name
    )

    # All 5 annotation files
    suffixes = [
        '_gtFine_color.png',
        '_gtFine_instanceIds.png',
        '_gtFine_labelIds.png',
        '_gtFine_labelTrainIds.png',
        '_gtFine_polygons.json'
    ]

    for suffix in suffixes:
        label_file = train_lbl_dir / city / f"{base_name}{suffix}"

        if label_file.exists():
            shutil.copy(
                label_file,
                test_lbl_dir / city / label_file.name
            )
        else:
            print(f"Missing: {label_file}")

print("✅ Test split with full annotations created successfully")