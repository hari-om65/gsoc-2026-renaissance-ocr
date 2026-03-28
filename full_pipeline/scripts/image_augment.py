import cv2
import numpy as np
import random
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def add_ink_fading(img, intensity=0.3):
    faded = img.astype(np.float32)
    faded = faded + intensity * 255
    faded = np.clip(faded, 0, 255).astype(np.uint8)
    return faded

def add_bleed_through(img, intensity=0.15):
    flipped = cv2.flip(img, 1)
    blended = cv2.addWeighted(img, 1.0, flipped, intensity, 0)
    return blended

def add_stains(img, num_stains=3):
    result = img.copy()
    h, w = result.shape[:2]
    for _ in range(num_stains):
        cx = random.randint(0, w-1)
        cy = random.randint(0, h-1)
        radius = random.randint(5, 30)
        color = random.randint(180, 230)
        cv2.circle(result, (cx, cy), radius, (color, color, color-20), -1)
    result = cv2.GaussianBlur(result, (5, 5), 0)
    blended = cv2.addWeighted(img, 0.7, result, 0.3, 0)
    return blended

def add_noise(img, intensity=15):
    noise = np.random.normal(0, intensity, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_skew(img, max_angle=2):
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    return rotated

def add_blur(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def add_contrast(img, alpha=1.3, beta=-20):
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def augment_line_image(img, difficulty="medium"):
    """Apply random augmentations based on difficulty level."""
    augmented = img.copy()
    
    if difficulty == "easy":
        augmented = add_noise(augmented, intensity=8)
        if random.random() > 0.5:
            augmented = add_skew(augmented, max_angle=1)
    
    elif difficulty == "medium":
        augmented = add_noise(augmented, intensity=12)
        augmented = add_ink_fading(augmented, intensity=0.15)
        if random.random() > 0.5:
            augmented = add_skew(augmented, max_angle=2)
        if random.random() > 0.5:
            augmented = add_blur(augmented, ksize=3)
    
    elif difficulty == "hard":
        augmented = add_noise(augmented, intensity=20)
        augmented = add_ink_fading(augmented, intensity=0.25)
        augmented = add_bleed_through(augmented, intensity=0.15)
        augmented = add_stains(augmented, num_stains=2)
        augmented = add_skew(augmented, max_angle=3)
        if random.random() > 0.5:
            augmented = add_blur(augmented, ksize=5)
    
    return augmented

# ── Demo on real line crops ──────────────────────────────────
crops_dir = Path.home() / "ocr_project/outputs/line_crops/print"
out_dir = Path.home() / "ocr_project/outputs/augmented_samples"
out_dir.mkdir(exist_ok=True)

# Find some good crops
sample_crops = []
for source in sorted(crops_dir.iterdir()):
    if not source.is_dir():
        continue
    for page_dir in sorted(source.iterdir()):
        if not page_dir.is_dir():
            continue
        crops = sorted(page_dir.glob("line_*.png"))
        for crop in crops[:2]:
            img = cv2.imread(str(crop))
            if img is not None and img.shape[0] > 20 and img.shape[1] > 100:
                sample_crops.append((crop, img))
        if len(sample_crops) >= 4:
            break
    if len(sample_crops) >= 4:
        break

print(f"Found {len(sample_crops)} sample crops for augmentation demo")

# Create comparison visualization
for idx, (crop_path, original) in enumerate(sample_crops[:4]):
    fig, axes = plt.subplots(1, 4, figsize=(24, 3))
    
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis('off')
    
    for i, diff in enumerate(["easy", "medium", "hard"]):
        aug = augment_line_image(original, difficulty=diff)
        axes[i+1].imshow(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
        axes[i+1].set_title(f"Augmented ({diff})", fontsize=10)
        axes[i+1].axis('off')
        
        # Save augmented versions
        aug_path = out_dir / f"{crop_path.stem}_aug_{diff}.png"
        cv2.imwrite(str(aug_path), aug)
    
    fig_path = out_dir / f"comparison_{idx}.png"
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

# ── Generate augmented training set ──────────────────────────
print("\nGenerating augmented training images...")
aug_train_dir = out_dir / "augmented_crops"
aug_train_dir.mkdir(exist_ok=True)

total_augmented = 0
for source in sorted(crops_dir.iterdir()):
    if not source.is_dir():
        continue
    for page_dir in sorted(source.iterdir()):
        if not page_dir.is_dir():
            continue
        for crop_path in sorted(page_dir.glob("line_*.png"))[:10]:
            img = cv2.imread(str(crop_path))
            if img is None or img.shape[0] < 15:
                continue
            for diff in ["easy", "medium", "hard"]:
                aug = augment_line_image(img, difficulty=diff)
                aug_name = f"{source.name}_{page_dir.name}_{crop_path.stem}_{diff}.png"
                cv2.imwrite(str(aug_train_dir / aug_name), aug)
                total_augmented += 1

print(f"Generated {total_augmented} augmented images")
print(f"Saved to: {aug_train_dir}")
print("\nStep 5.3 Complete!")
