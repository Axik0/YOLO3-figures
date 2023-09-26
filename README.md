# YOLO3-figures

## Data generation
My algorithm in generation.py creates RGB 256,256,3 HWC images with various 2D-shapes

* random max size 25..150px
* random colour
* random place
* random angle (for symmetric)
* random amount of figures 1..5
* random figure shape: Circle, Rhombus, Rectangle, Triangle, Polygon(Hexagon or any symmetric)

I've generated 10K of such Images
![Screenshot 2023-09-12 170355](https://github.com/Axik0/YOLO3-figures/assets/100946131/d312f430-d323-4dde-ab0a-865ae0942e66)

I store local paths to images as keys within json data object for convenience, 

image description includes shape_class(string), bounding box (bb_center_x, bb_center_y), (bb_half_width, bb_half_height) for all figures
## Dataset preparation
### Image transformations: 
I use Albumentations as my transformation/augmentation framework
1. Since YOLO image size is 416, I have to resize and convert my 256,256,3 uint8 hwc PIL images to torch dloat chw tensors by default, this transform DEFAULT_TR is built-in
2. To help gradient flow, I use Normalization with actual statistics collected from all 10K images
   (per-channel mean=[0.641, 0.612, 0.596], std=[0.115, 0.11, 0.11] also built-in constants)
4. Since my dataset is already quite random, i don't think i need any augmentations, but still apply ColorJitter and RandomHorizontalFlip to train dataset. There's no technical problem to use any other augmentations since A. transforms images and bboxes altogether

Unfortunately I can't apply them at once as it requires more than 8Gb of RAM, thus all transformations are applied on the fly (in __getitem__)

### Target processing:
Yolo has 3 outputs and requires specific targets, I prepare 3 tensors each shaped 3*S*S*6, where last dimension is (object presence, bbox(4 local coordinates), class_id)

# TO BE CONTINUED
