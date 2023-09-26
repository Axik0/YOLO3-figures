# YOLO3-figures
## Data generation
My algorithm in generation.py creates RGB 256,256,3 HWC images with up to 5 various 2D-shapes (random max size 25..150px, random colour, random place, random angle)
For now, allowed shapes are Circle, Rhombus, Rectangle, Triangle, Polygon(Hexagon or any symmetric) and I've generated 10K of such Images
![Screenshot 2023-09-12 170355](https://github.com/Axik0/YOLO3-figures/assets/100946131/d312f430-d323-4dde-ab0a-865ae0942e66)
Their description includes shape_class(string), bounding box (bb_center_x, bb_center_y), (bb_half_width, bb_half_height) for all figures
I also uses local paths to images as keys within json data object for convenience
## Dataset preparation
When I load dataset collected statistics (per-channel mean=[0.641, 0.612, 0.596], std=[0.115, 0.11, 0.11]) from all 10K images for Normalization transform. 
Recalculation is disabled by default and used built-
