# &nbsp;📂 Dataset Setup for VQA and Reasoning (ALBEF-based Project)

# 

# All datasets are stored \*\*in Google Drive\*\*, not in this repository.

# Create a folder structure like:



MyDrive/data/

│

├── visual\_genome/

├── coco/

├── vqa/

├── nlvr2/

├── snli-ve/

├── flickr30k/

└── refcoco\_plus/





# 

# ---

# 

# \## 1️⃣ Visual Genome (v1.2)

# 

# \*\*Purpose:\*\* region-level grounding and question–answer pretraining.

# 

# \*\*Download page:\*\* \[Visual Genome Downloads](https://visualgenome.org/api/v0/api\_home.html)

# 

# \*\*Download these files (v1.2):\*\*

# \- Images part 1 (VG\_100K) — 9.2 GB

# \- Images part 2 (VG\_100K\_2) — 5.47 GB

# \- image\_data.json — 17.6 MB

# \- region\_descriptions.json — 712 MB

# \- question\_answers.json — 803 MB

# \- objects.json — 413 MB

# \- attributes.json — 462 MB

# \- relationships.json — 709 MB

# 

# \*\*Skip:\*\* synsets, aliases, scene graphs.

# 

# \*\*Folder layout:\*\*



data/visual\_genome/

├── images/VG\_100K/

├── images/VG\_100K\_2/

├── image\_data.json

├── region\_descriptions.json

├── question\_answers.json

├── objects.json

├── attributes.json

└── relationships.json





---



\## 2️⃣ COCO Captions (2014 or 2017)



\*\*Purpose:\*\* image–caption pretraining and retrieval.



\*\*Download:\*\* \[COCO Dataset](https://cocodataset.org/#download)



\- Images: `train2014`, `val2014` (≈ 20 GB)

\- Captions:

  - `captions\\\_train2014.json`

  - `captions\\\_val2014.json`

data/coco/

├── train2014/

├── val2014/

├── captions\_train2014.json

└── captions\_val2014.json





---



\## 3️⃣ VQA v2.0



\*\*Purpose:\*\* fine-tuning and evaluating the Visual Question Answering task.



\*\*Download:\*\* \[VQA v2 Downloads](https://visualqa.org/download.html)



Files:

\- `v2\\\_Questions\\\_Train\\\_mscoco.zip`

\- `v2\\\_Questions\\\_Val\\\_mscoco.zip`

\- `v2\\\_Annotations\\\_Train\\\_mscoco.zip`

\- `v2\\\_Annotations\\\_Val\\\_mscoco.zip`



\*\*Uses the same COCO images.\*\*

├── v2\_OpenEnded\_mscoco\_train2014\_questions.json

├── v2\_mscoco\_train2014\_annotations.json

├── v2\_OpenEnded\_mscoco\_val2014\_questions.json

└── v2\_mscoco\_val2014\_annotations.json





---



\## 4️⃣ NLVR2



\*\*Purpose:\*\* visual reasoning (two images + text).



\*\*Download:\*\* \[NLVR2 Dataset](https://lil.nlp.cornell.edu/nlvr/)



Files:

\- `train.zip`, `dev.zip`, `test.zip`

Extract images and JSON annotations.

data/nlvr2/

├── images/

├── train.json

├── dev.json

└── test.json





---



\## 5️⃣ SNLI-VE



\*\*Purpose:\*\* visual-textual entailment.



\*\*Download:\*\* \[SNLI-VE GitHub](https://github.com/necla-ml/SNLI-VE)



Files:

\- `snli\\\_ve\\\_train.jsonl`

\- `snli\\\_ve\\\_dev.jsonl`

\- `snli\\\_ve\\\_test.jsonl`





---



\## 5️⃣ SNLI-VE



\*\*Purpose:\*\* visual-textual entailment.



\*\*Download:\*\* \[SNLI-VE GitHub](https://github.com/necla-ml/SNLI-VE)



Files:

\- `snli\\\_ve\\\_train.jsonl`

\- `snli\\\_ve\\\_dev.jsonl`

\- `snli\\\_ve\\\_test.jsonl`

data/snli-ve/

├── train.jsonl

├── dev.jsonl

└── test.jsonl





---



\## 6️⃣ Flickr30k



\*\*Purpose:\*\* image–text retrieval and grounding evaluation.



\*\*Download:\*\* \[Flickr30k Entities](http://shannon.cs.illinois.edu/DenotationGraph/)



Files:

\- `flickr30k\\\_images/`

\- `flickr30k\\\_captions.json`

data/flickr30k/

├── images/

└── captions.json





---



\## 7️⃣ RefCOCO+



\*\*Purpose:\*\* referring-expression / phrase-grounding evaluation.



\*\*Download:\*\* \[RefCOCO+ Repo](https://github.com/lichengunc/refer)



Files:

\- `refcoco+\\\_annotations.json`

\- COCO images (reuse from `data/coco`)



data/refcoco\_plus/

├── annotations.json

└── images/ ← link or copy to COCO images







---



\## ⚙️ Notes



\- Keep large datasets on Drive and mount Drive in Colab:

  ```python

  from google.colab import drive

  drive.mount('/content/drive')

  data\_root = "/content/drive/MyDrive/data"



