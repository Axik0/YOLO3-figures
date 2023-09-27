# YOLO3-figures

## Data generation
My algorithm in generation.py creates RGB 256,256,3 HWC images with various 2D-shapes

* random max size 25..150px
* random colour
* random place
* random angle 0..45 (for symmetric)
* random amount 1..5 of non-intersecting figures
* random figure shape: Circle, Rhombus, Rectangle, Triangle, Polygon(Hexagon or any symmetric)

I've generated 10K of such Images
![Screenshot 2023-09-12 170355](https://github.com/Axik0/YOLO3-figures/assets/100946131/d312f430-d323-4dde-ab0a-865ae0942e66)

Image description includes (for all figures):
- shape_class(string), 
- bounding box params (bb_center_x, bb_center_y), (bb_half_width, bb_half_height)

I store local paths to images as keys within json data object for loading convenience

## Dataset preparation
### Image transformations: 
I use Albumentations as my transformation/augmentation framework
1. Since YOLO image size is 416, I have to resize and convert my 256,256,3 uint8 hwc PIL images to torch dloat chw tensors by default, this transform DEFAULT_TR is built-in
2. To help gradient flow, I use Normalization with actual statistics collected from all 10K images
   (per-channel mean=[0.641, 0.612, 0.596], std=[0.115, 0.11, 0.11] are also built-in constants)
4. Since my data is already quite diverse, I don't think I need any augmentations, but still apply ColorJitter and RandomHorizontalFlip to train dataset. There's no technical problem to use any other augmentations since A. transforms images and bboxes altogether

Unfortunately I can't apply them at once as it requires more than 8Gb of RAM, thus all transformations are applied on the fly (in __getitem__)

### Target processing:

Yolo makes predictions at 3 scales at once thus has 3 outputs and requires specific targets, I prepare 3 tensors each shaped (3, S, S, 6), where last dimension is:
- object presence score 
- bbox(4 local coordinates within a cell of grid S)
- class_id

It's easier to think of that as coarse (up to cell) then fine (within a cell) coordinate system for all objects on image

People say pretty much confusing things like "each cell predicts 1 object" but in fact, targets for each image are independent and even then there isn't any 1:1 correspondence. 
It varies from 0:1 to 3:1 as our objects are supposed to be detected on all scales at once, independently and some object might not be included into target at all if that cell is 'full'.
#### Anchors
Each cell has 3 pre-defined bboxes aka anchors at each of 3 scales. 
They are priors, represent typical aspect ratios and sizes within a dataset. As my dataset is vastly different from COCO, I can't use default and have to get custom ones. I apply clustering algorithm ~Kmeans with IoU to all bboxes to retrieve 9 cluster centers, sorted by size (w+h)

In theory, Yolov3 allows up to 3 objects 'taking up' a same cell but for now, max amount of objects is 5 and they (even their bboxes) aren't intersecting with each other by construction, therefore our target tensors are quite sparse and never 'exhausted'

## Prediction
To achieve better gradient flow, Yolo model doesn't output coordinates of bbox, instead it controls (parameters of) transformation (defined within loss function) of anchor to bbox

We use special combo-loss function of 4 components, weighed differently

Object score is not just 0/1, should be probability that reflects model's confidence (in bbox) 
Therefore we compare prediction bbox with target so that model could learn IoU as score
For yet unknown reason, IoU calculation is detached from the current graph here

BBox loss is in fact 3 orders of magnitude of other 3 parts, that's why I don't scale it 10x more as in original paper (set higher weights for presence and absence losses instead)

# TO BE CONTINUED
