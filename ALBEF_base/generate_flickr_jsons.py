import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your results.csv
csv_path = r"C:\Users\TL1\Efficient-ALBEF-Alignment\flickr30k_images\results.csv"

# Folder to save JSONs
output_dir = r"C:\Users\TL1\Efficient-ALBEF-Alignment\ALBEF_base\data"
os.makedirs(output_dir, exist_ok=True)

# Load CSV with pipe separator
df = pd.read_csv(csv_path, sep='|', names=['image_name', 'comment_number', 'comment'], header=0, encoding='utf-8')

# Strip whitespace from column values
df['image_name'] = df['image_name'].str.strip()
df['comment'] = df['comment'].str.strip()

# Create list of (image, caption) pairs
data = []
for _, row in df.iterrows():
    img = row['image_name']
    caption = row['comment']
    data.append({'image': img, 'caption': caption})

# Use Karpathy split ratios (train 29k, val 1k, test 1k)
# Shuffle first
data = pd.DataFrame(data)
unique_images = data['image'].unique()
train_imgs, temp_imgs = train_test_split(unique_images, test_size=2000, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=1000, random_state=42)

def filter_by_images(img_list):
    return [row for row in data.to_dict('records') if row['image'] in img_list]

train_data = filter_by_images(train_imgs)
val_data = filter_by_images(val_imgs)
test_data = filter_by_images(test_imgs)

# Save JSONs
with open(os.path.join(output_dir, 'flickr30k_train.json'), 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4)
with open(os.path.join(output_dir, 'flickr30k_val.json'), 'w', encoding='utf-8') as f:
    json.dump(val_data, f, indent=4)
with open(os.path.join(output_dir, 'flickr30k_test.json'), 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4)

print(" JSON files created successfully in:", output_dir)
