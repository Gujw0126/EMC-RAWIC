import os
import numpy as np
import rawpy
import glob
from tqdm import tqdm

#split NUS train dataset 
def save_raw_patches(root_dir, save_dir, patch_size=512, stride=512):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    extensions = ('*.dng', '*.ARW', '*.CR2', '*.NEF', '*.ORF', '*.raf','*.SRW')
    raw_paths = []
    for ext in extensions:
        raw_paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))

    print(f"find {len(raw_paths)} RAW images.Splitting...")
    patch_idx = 0
    for img_path in tqdm(raw_paths, desc="processing:"):
        try:
            with rawpy.imread(img_path) as raw:
                raw_data = raw.raw_image.copy()
                black_level = np.uint16(raw.black_level_per_channel[0])
                white_level = np.uint16(raw.camera_white_level_per_channel[0])
                white_level=4095
                
                h, w = raw_data.shape
            
                for y in range(0, h - patch_size + 1, stride):
                    for x in range(0, w - patch_size + 1, stride): 
                        if y % 2 != 0 or x % 2 != 0:
                            continue
                        patch = raw_data[y:y+patch_size, x:x+patch_size].copy()
                        base_name = os.path.basename(img_path).split('.')[0]
                        filename = f"{base_name}_idx{patch_idx:06d}_y{y}_x{x}.npz"
                        
                        np.savez_compressed(
                            os.path.join(save_dir, filename),
                            data=patch,
                            black=black_level,
                            white=white_level
                        )
                        patch_idx += 1
                        
        except Exception as e:
            print(f"\nfilename: {img_path}: {e}")

    print(f"\nprocess over！")
   

if __name__ == "__main__":
    SOURCE_ROOT = "xxx" 
    TARGET_DIR = "xxx"
    
    save_raw_patches(
        root_dir=SOURCE_ROOT,
        save_dir=TARGET_DIR,
        patch_size=512, 
        stride=512       
    )
