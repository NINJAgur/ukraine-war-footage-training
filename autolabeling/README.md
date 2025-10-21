# YOLO Dataset Creation with GroundingDINO

Required Python Packages:

`pip install opencv-python groundingdino-py`

### GroundingDINO Model Files:

Configuration file: GroundingDINO_SwinT_OGC.py
Checkpoint file: groundingdino_swint_ogc.pth
Download these files from the GroundingDINO repository or use pre-trained weights provided by the authors.

```console
wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
```

### Input Images:

A folder containing images (supported formats: .jpg, .jpeg, .png) to be annotated.


### Setup

Prepare Input Folder:

Place your images in a folder (e.g., litter_on_forest_floor/litter).


### Download GroundingDINO Files:

Place GroundingDINO_SwinT_OGC.py and groundingdino_swint_ogc.pth in the same directory as the script or specify their paths when running the script.


### Usage
The script takes three required command-line arguments:

`--input_folder`: Path to the folder containing images to annotate.
`--text_prompt`: Comma-separated list of object classes to detect (e.g., plastic bottle, can, wrapper, bag, trash).
`--output_path`: Path where the YOLO dataset will be created.

Optional arguments:

`--config_path`: Path to GroundingDINO config file (default: GroundingDINO_SwinT_OGC.py).
`--checkpoint_path`: Path to GroundingDINO checkpoint file (default: groundingdino_swint_ogc.pth).
`--box_threshold`: Confidence threshold for bounding boxes (default: 0.35).
`--text_threshold`: Confidence threshold for text phrases (default: 0.25).

Example Command
```
python annotate_images_for_yolo.py \
  --input_folder "litter_on_forest_floor/litter" \
  --text_prompt "plastic bottle, can, wrapper, bag, trash" \
  --output_path "yolo_dataset"
```

### Output Structure
The output folder (`yolo_dataset` in the example) will have the following structure:
```
yolo_dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── labels/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
├── data.yaml
```

Example `data.yaml`
path: /absolute/path/to/yolo_dataset
train: train/images
val: train/images
nc: 5
names: ['plastic bottle', 'can', 'wrapper', 'bag', 'trash']
