import json, shutil, os

small_json = "C:/Users/TL1/Efficient-ALBEF-Alignment/ALBEF_base/data/flickr30k_val_small.json"   # your filtered JSON file
full_img_dir = "C:/Users/TL1/Efficient-ALBEF-Alignment/flickr30k_images/flickr30k_images"
small_img_dir = "C:/Users/TL1/Efficient-ALBEF-Alignment/flickr30k_images/flickr30k_small_val"

os.makedirs(small_img_dir, exist_ok=True)

anns = json.load(open(small_json))

image_list = list({ann["image"] for ann in anns})  # unique images

print("Copying", len(image_list), "images...")

for img in image_list:
    src = os.path.join(full_img_dir, img)
    dst = os.path.join(small_img_dir, img)
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print("Missing:", img)

print("Done.")
