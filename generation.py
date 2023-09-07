import os
import json
import inspect

# i won't use numpy for a task that simple intentionally
import random
import statistics
from math import pi, sin, cos, sqrt
from itertools import accumulate, product
# drawing framework
from PIL import Image, ImageDraw, ImageColor
# template class for a dataset (makes our dataset compatible with torchvision)
from torchvision.datasets import VisionDataset
from torchvision.io import read_image

RSEED = 42
PATH = './'
FNAME = 'pics'
DNAME = 'data.json'

SIZE = 256
# 25++ because inscribed objects might be smaller
THR = 25 + 20
PARTS = 5
MARGIN = 3


def get_data(json_object_path):
    with open(json_object_path, 'r') as f:
        res = json.load(f)
    return res


# def load_image(image_path):
#     with open(image_path, 'rb') as p:
#         return p


def rand_split(n, k: int, l_bound=0):
    """recursive random split, performs (correlated) random split of n
     onto k parts, outputs list of k spacers > l_bound that sum up to n"""
    unc = [random.random() for _ in range(k)]
    raw_spaces = list(map(lambda x: round(n * x / sum(unc)), unc))
    m = max(raw_spaces)
    err = abs(n - sum(raw_spaces))
    norm, redo, maxes = [], [], []
    for s in raw_spaces:
        # put away all elements below threshold (& max element)
        if s == m:
            # deal with possibly multiple maxes
            if not maxes:
                # adjust one of maxes to get rid of error
                m_ = m - err
                err = 0
            else:
                m_ = m
            maxes.append(m_)
        elif s > l_bound:
            norm.append(s)
        else:
            redo.append(s)
        # print(norm, maxes, redo)
    if redo:
        # add up available space
        n_redo = sum(redo) + sum(maxes)
        c_redo = len(redo) + len(maxes)
        # redistribute remaining space (& max element)
        if n_redo / c_redo <= l_bound + 5:
            # prevent recursion overflow for wasted leftovers
            spaces = rand_split(n, k, l_bound)
        else:
            done = rand_split(n_redo, c_redo, l_bound)
            spaces = norm + done
        # print(raw_spaces, norm, redo, maxes)
    else:
        spaces = norm + maxes
    assert sum(spaces) <= n, f'Result {spaces} has round-off error(s)'
    assert len(spaces) == k, f'Result {spaces} has wrong length'
    return spaces


# try:
#     for i in range(10000):
#         rand_split(256, 5, THR)
#     print('10K-pass')
# except AssertionError:
#     print('function behaves wrong')

def get_centers(n=SIZE, k=PARTS - 2, l_bound=THR, mute=False):
    """convenient function that transforms exact parameters from just spacers"""
    assert k > 2, 'wrong PARTS quantity'
    spacers = rand_split(n, k, l_bound)
    delimiters = [0] + list(accumulate(spacers))[:-1] + [256]
    centers = [round(p + (f - p) / 2) for f, p in zip(delimiters[1:], delimiters[:-1])]
    half_sizes = [s / 2 for s in spacers]
    if not mute:
        print(f'We have {len(spacers)} intervals allocated as {delimiters}, their centers are {centers}')
    return centers, half_sizes


def rc_parts(xy_list, wh_list):
    """choose up to 5 different boxes"""
    # random choice without repetitions
    idx = list(range(len(xy_list)))
    random.shuffle(idx)
    # random quantity from 1..PARTS
    q = random.randint(1, PARTS)
    return [(xy_list[i], wh_list[i]) for i in idx[:q]]


class Figure:
    def __init__(self, center: tuple, half_size: tuple):
        self.shape = None
        self.x, self.y = center
        self.half_w, self.half_h = half_size
        self.bbox = None
        # local coordinates of vertices, relative to the center
        self.ve_loc = (-self.half_w, - self.half_h), (self.half_w, self.half_h)


class BBox(Figure):
    def __init__(self, center, half_size):
        """rectangle w/ those parameters"""
        super().__init__(center, half_size)
        self.shape = self.__class__.__name__
        self.wh = [2 * d for d in (self.half_w, self.half_h)]
        self.area = self.wh[0] * self.wh[1]
        # alternative minmax vertex representation
        self.ve_min = round(self.x - self.half_w), round(self.y - self.half_h)
        self.ve_max = round(self.x + self.half_w) - 1, round(self.y + self.half_h) - 1
        # global coordinates of farthest vertices
        self.ve = self.ve_min, self.ve_max
        # local coordinates of vertices, relative to the center
        self.ve_loc = (-self.half_w, - self.half_h), (self.half_w, self.half_h)

    def __repr__(self):
        image = Image.new(mode='RGB', size=(SIZE, SIZE), color='white')
        canvas = ImageDraw.Draw(image)
        canvas.rectangle(self.ve, fill='black')
        canvas.point(self.ve[0], 'red')
        canvas.point(self.ve[1], 'red')
        image.show()
        return f'{self.shape} that starts at {self.ve_min} with width {self.wh[0]}, height {self.wh[1]} and area {self.area}'

    def __add__(self, other):
        """combines adjacent rectangles in a row or column"""
        dx, dy = (self.x - other.x), (self.y - other.y)
        assert dx * dy == 0, 'not inline!'
        # 0 - vertical border case or both vertical borders coincide
        th = round(abs(self.x - other.x - self.half_w) - other.half_w)
        # 0 - horizontal border case or both horizontal borders coincide
        tw = round(abs(self.y - other.y - self.half_h) - other.half_h)
        assert tw + th == 0, f'not adjacent!, {tw, th}'
        # assert (self.half_w - other.half_w)*(self.half_h + other.half_h) == 0, f'not rectagonal'
        center = round((self.x + other.x) / 2), round((self.y + other.y) / 2)
        nhw, nhh = self.half_h, self.half_w
        if (not th) and dx:
            nhw = (self.half_w + other.half_w)
        elif (not tw) and dy:
            nhh = (self.half_h + other.half_h)
        else:
            print(f'wrong allocation,{tw, th})')
        half_size = nhw, nhh
        res = BBox(center, half_size)
        return res

    def get_aoi(self, other):
        """calculates (area of) intersection of two rectangles"""
        assert isinstance(other, BBox), f'object {other} is not an instance of {self.__class__.__name__}'
        # nearest max vertex - farthest min vertex
        dx = min(self.ve_max[0], other.ve_max[0]) - max(self.ve_min[0], other.ve_min[0])
        dy = min(self.ve_max[1], other.ve_max[1]) - max(self.ve_min[1], other.ve_min[1])
        if dx >= 0 and dy >= 0:
            return dx * dy
        else:
            print('no intersection')

    def get_iou(self, other):
        """produces IoU ratio (percentage)"""
        assert isinstance(other, BBox), f'object {other} is not an instance of {self.shape}'
        aoi = self.get_aoi(other)
        if aoi:
            aou = self.area + other.area - aoi
            iou = round(aoi / aou, 2) * 100
        else:
            iou = 0
        return iou


class Triangle(Figure):
    def __init__(self, center, half_size, std_threshold=0.7):
        super().__init__(center, half_size)
        self.shape = self.__class__.__name__
        half_size = list(map(round, half_size))
        self.bbox_ = BBox(center, half_size)
        shift_x = list(range(-half_size[0] + 2 * MARGIN, half_size[0] - 2 * MARGIN))
        shift_y = list(range(-half_size[1] + 2 * MARGIN, half_size[1] - 2 * MARGIN))
        random.shuffle(shift_x)
        random.shuffle(shift_y)
        # prevent slim triangles as it's hard to distinguish those
        std_x, std_y = 0, 0
        while std_x < std_threshold * half_size[0] or std_y < std_threshold * half_size[1]:
            random.shuffle(shift_x)
            random.shuffle(shift_y)
            std_x, std_y = statistics.stdev(shift_x[:3]), statistics.stdev(shift_y[:3])
        x, y = [self.x + sx for sx in shift_x[:3]], [self.y + sy for sy in shift_y[:3]]
        self.ve = list(zip(x, y))
        ve_min = min(x), min(y)
        ve_max = max(x), max(y)
        bc = round((ve_max[0] + ve_min[0]) / 2), round((ve_max[1] + ve_min[1]) / 2)
        hs = list((d[1] - d[0] + MARGIN) / 2 + 1 for d in zip(ve_min, ve_max))
        self.bbox = BBox(bc, hs)

    def __repr__(self):
        image = Image.new(mode='RGB', size=(SIZE, SIZE), color='white')
        canvas = ImageDraw.Draw(image)
        canvas.polygon(self.ve, fill='black')
        canvas.rectangle(self.bbox_.ve, outline='red')
        canvas.rectangle(self.bbox.ve, outline='green')
        image.show()
        return f'{self.shape} bounded by {self.bbox.ve}'

    def draw(self, canvas, colour):
        return canvas.polygon(self.ve, fill=colour)


class Rhombus(Figure):
    def __init__(self, center, half_size, s=0.9):
        """random size within bounding box, smaller s -- thinner rhombus """
        super().__init__(center, half_size)
        self.shape = self.__class__.__name__
        self.bbox_ = BBox(center, half_size)
        # add some margin
        # self.bbox = BBox(center, (hs - MARGIN for hs in half_size))
        self.max_diag = (min(half_size) - MARGIN) / sqrt(2)
        self.bbox = BBox(center, (self.max_diag, self.max_diag))
        # we assume that we use ve_min, ve_max points as a frame
        # diagonal min-max -dy*x+dx*y + x1y2-x2y1=0 w/ normal (-dy,dx),
        dx = self.bbox.ve_max[0] - self.bbox.ve_min[0]
        dy = self.bbox.ve_max[1] - self.bbox.ve_min[1]
        # let's find an orthogonal line, i.e. w/ normal (dx,dy) and ic
        line = (dx, dy, -(dx * self.x + dy * self.y))

        def cross(nni, xs):
            # n1*x+n2*y+intercept=0 for all xs --> ys
            assert nni[1] != 0, 'wrong line, dna zerodiv'
            return list(map(lambda x: -round((nni[0] * x + nni[2]) / nni[1]), xs))

        x_lim = self.bbox.x + MARGIN, round(self.bbox.ve_max[0]) - MARGIN
        side_x = random.randint(x_lim[0] + round(s * (x_lim[1] - x_lim[0] - 2 * MARGIN)) - MARGIN, x_lim[1])
        # invert side_x (relative to center)
        side_xi = self.x - (side_x - self.x)
        side_xs = (side_xi, side_x)
        # then corresponding ys
        side_ys = cross(line, side_xs)
        self.side_pts = [p for p in zip(side_xs, side_ys)]
        self.far_pts = [self.bbox.ve_min, self.bbox.ve_max]
        # we have to sort this array to fill in colours properly
        # prevent connections between opposite vertices (diagonal) => interleaving arrays
        self.ve = [v for pair in zip(self.side_pts, self.far_pts) for v in pair]

    def __repr__(self):
        image = Image.new(mode='RGB', size=(SIZE, SIZE), color='white')
        canvas = ImageDraw.Draw(image)
        canvas.polygon(self.ve, fill='black')
        canvas.point(self.side_pts[0], 'red')
        canvas.point(self.side_pts[1], 'red')
        canvas.rectangle(self.bbox.ve, outline='green')
        canvas.rectangle(self.bbox_.ve, outline='red')
        image.show()
        return f'{self.shape} with vertices {self.ve} bounded by {self.bbox.ve}'

    def draw(self, canvas, colour):
        return canvas.polygon(self.ve, fill=colour)


class Circle(Figure):
    def __init__(self, center, half_size, s=0.9):
        super().__init__(center, half_size)
        self.shape = self.__class__.__name__
        self.bbox_ = BBox(center, half_size)
        self.max_radius = min(half_size) - MARGIN
        self.radius = random.uniform(0.7, s) * self.max_radius
        # crop to square w/ margin
        self.bbox = BBox(center, (self.radius, self.radius))
        self.ve = [(self.x - self.radius, self.y - self.radius), (self.x + self.radius, self.y + self.radius)]

    def __repr__(self):
        image = Image.new(mode='RGB', size=(SIZE, SIZE), color='white')
        canvas = ImageDraw.Draw(image)
        canvas.ellipse(self.ve, fill='black')
        canvas.rectangle(self.bbox.ve, outline='green')
        canvas.rectangle(self.bbox_.ve, outline='red')
        canvas.point((self.x, self.y), fill='white')
        image.show()
        return f'{self.shape} with radius {self.radius} bounded by green box ~ {self.bbox.ve}'

    def draw(self, canvas, colour):
        return canvas.ellipse(self.ve, fill=colour)


class Polygon(Figure):
    def __init__(self, center, half_size, ratio=0.7, nv=6, angle=0):
        """works for any symmetric polygon, not just Hexagon, let it be just 3 possible shapes
        ratio=0..1 corresponds to fill-in ratio of max possible circle within start bbox,
         nv is an amount of vertices (6 for Hexagon), angle ~ starting angle (counterclockwise)"""
        super().__init__(center, half_size)
        names = {4: 'Square', 5: 'Pentagon', 6: 'Hexagon'}
        assert nv in names.keys(), 'wrong amount of vertices'
        self.shape = names[nv]
        # default bbox, excessive, might not fit precisely to the shape (but still fine)
        self.bbox_ = BBox(center, half_size)
        # max radius of a circle that could be inscribed into this box - MARGIN
        self.max_radius = min(half_size) - MARGIN
        # polygon final size
        self.radius = ratio * self.max_radius
        # crop to square w/ margin (lower tolerance bbox)
        self.bbox = BBox(center, (self.radius, self.radius))

        def split(n, k, start=0):
            """get k sequential! parts starting from some value"""
            assert n % k == 0, f'{n} is not divisible by {k}, stop'
            return list(range(start, start + n, n // k))

        self.angles_d = split(360, nv, angle)
        # convert to radians
        self.angles_r = list(map(lambda a: a * pi / 180, self.angles_d))
        # embed -a to make this comfy visually, PIL uses different coordinate system => clockwise angle count
        self.ve = [(self.x + self.radius * cos(-a), self.y + self.radius * sin(-a)) for a in self.angles_r]

    def __repr__(self):
        image = Image.new(mode='RGB', size=(SIZE, SIZE), color='white')
        canvas = ImageDraw.Draw(image)
        canvas.polygon(self.ve, fill='black')
        canvas.rectangle(self.bbox.ve, outline='green')
        canvas.rectangle(self.bbox_.ve, outline='red')
        canvas.line(((self.x, self.y), self.ve[0]), fill='blue')
        canvas.point((self.x, self.y), fill='white')
        image.show()
        return f'{self.shape} with angles {self.angles_d} and radius {self.radius} bounded by green box ~ {self.bbox.ve}'

    def draw(self, canvas, colour):
        return canvas.polygon(self.ve, fill=colour)


class Rectangle(Polygon):
    def __init__(self, center, half_size, ratio=0.7, angle=0, max_sqrnss=0.8):
        super().__init__(center, half_size, ratio=ratio, angle=angle)
        """we are going to inscribe this rectangle into max possible circle of Polygon class, 
        and set up with just 2 angles(start, add up to 90deg to get 2nd vertex, then reflect to get the remaining two)
        parameter squareness=0...1 yields a square when 1, supposed to be random"""
        self.shape = self.__class__.__name__
        self.sqrnss = random.uniform(0.5, max_sqrnss)
        self.angular_delta = self.sqrnss * 90

        # set up vertices counterclockwise, convert to radians, overwrite default polygon attributes (Hexagon
        self.angles_d = [angle, angle + self.angular_delta, 180 + angle, 180 + angle + self.angular_delta]
        self.angles_r = list(map(lambda a: a * pi / 180, self.angles_d))
        self.ve = [(self.x + self.radius * cos(-a), self.y + self.radius * sin(-a)) for a in self.angles_r]
        # update bounding box
        new_bb_min = min(self.ve, key=lambda t: t[0])[0] - MARGIN, min(self.ve, key=lambda t: t[1])[1] - MARGIN
        new_bb_max = max(self.ve, key=lambda t: t[0])[0] + MARGIN, max(self.ve, key=lambda t: t[1])[1] + MARGIN
        new_wh = ((new_bb_max[0] - new_bb_min[0]) / 2, (new_bb_max[1] - new_bb_min[1]) / 2)
        self.bbox = BBox(center, new_wh)


def draw_bounds(res):
    image = Image.new(mode='RGB', size=(SIZE, SIZE), color='white')
    canvas = ImageDraw.Draw(image)
    for b in res:
        b_ = b[0], (map(lambda d: d - MARGIN, b[1]))
        box = BBox(*b_)
        canvas.rectangle(box.ve, fill='black')
        canvas.point(b[0], fill='white')
    image.show()


def draw_shapes(bboxes_list, mute=False):
    # random colour palette (6) from a default cmap
    colours_dict = ImageColor.colormap
    c_names = list(colours_dict.keys())
    # random.seed(RSEED)
    random.shuffle(c_names)
    palette = [colours_dict[name] for name in c_names[:6]]
    image = Image.new(mode='RGB', size=(SIZE, SIZE), color=palette[0])
    canvas = ImageDraw.Draw(image)
    figures = []
    sizes = []
    for b in bboxes_list:
        # random.seed(RSEED)
        sh_id = random.randint(1, 5)
        # random colour (of 140)
        colour_ch = palette[sh_id]
        # random shape (of 5)
        class_ch = id_to_class[sh_id - 1]
        # explore more options to randomize if possible (before instantiation)
        params = set(inspect.signature(class_ch).parameters)
        extra_params = params - {'center', 'half_size'}
        if len(extra_params) > 1:
            # otherwise they're Rhombus and Triangle, already quite random and depend on just the bbox
            # choose 'size'
            ratio_ch = random.uniform(0.9, 1)
            # rotate to an angle
            angle_ch = random.randrange(5, 90)
            obj = class_ch(*b, ratio=ratio_ch, angle=angle_ch)

        # set up an instance
        obj = class_ch(*b)
        obj.draw(canvas, colour_ch)

        size = (round(min(obj.bbox.wh)), round(max(obj.bbox.wh)))
        sizes.append(size)
        # check sizes
        if size[0] < 25 or size[1] > 150:
            raise ValueError(f'Figure {obj} exceeds (25...150) limits {size}')
        else:
            figures.append(obj)
    if not mute:
        print(f'Image contains: {len(figures)} figure(s): {[f.__class__.__name__ for f in figures]}')
    return image, figures


def tile(pil_img_list, amount=100):
    """concatenates ~amount images layout for visualization"""
    assert amount != 0, 'amount cant be just 0'
    assert len(pil_img_list) >= amount, 'we dont have that many images'
    side = int(sqrt(amount))
    tile_width = pil_img_list[0].width + MARGIN
    tile_height = pil_img_list[0].height + MARGIN
    new_width = tile_width * side
    new_height = tile_height * side
    img = Image.new(mode='RGB', size=(new_width, new_height))
    for i in range(side):
        # process row-wise
        for j in range(side):
            # concatenate horizontally
            img.paste(pil_img_list[j + i * side], (i * tile_height, j * tile_width))
    return img


def generate(n, root=PATH, folder_name=FNAME, data_name=DNAME, store=True):
    """generates n random 256*256 images within given limitations, random colours etc.,
    stores them all as pngs at img_path, their serialized description goes to data_path,"""
    data = {}
    img_to_show = []
    json_path = os.path.join(root, data_name)

    for i in range(n):
        # allocate Ox, Oy projections of possible rectangles
        x, y = get_centers(mute=True), get_centers(mute=True)
        # generate product of all xs, ys (all possible combinations without replacement)
        centers_xy, half_sizes_wh = list(product(x[0], y[0])), list(product(x[1], y[1]))
        choice = rc_parts(centers_xy, half_sizes_wh)
        # print(f'{len(centers_xy)} rectangles (and their shapes) in total, chosen {len(choice)}')
        try:
            result = draw_shapes(choice, mute=True)
        except ValueError:
            print('one missed, retried(once)')
            result = draw_shapes(choice)
        # result[0].show()
        img_to_show.append(result[0])
        # store picture
        local_img_path = os.path.join(folder_name, f'{i}.png')
        if store:
            abs_folder_path = os.path.join(root, folder_name)
            try:
                os.mkdir(abs_folder_path)
            except FileExistsError:
                pass
            finally:
                result[0].save(fp=os.path.join(root, local_img_path))
        # create descr ~ {img_path:[[fig1_type, bbox1_start, bbox1_wh],[fig1_type, bbox1_start, bbox1_wh],...], 1:...}
        description = list((f.shape, f.bbox.ve_min, f.bbox.wh) for f in result[1])
        data[local_img_path] = description
        if i % 200 == 0:
            print(i)
    # store its description
    if store:
        with open(json_path, 'w') as f:
            json.dump(data, f)
    print(f'{len(data)} images have been created {"and saved" if store else ""} successfully')
    return img_to_show


def load_dataset(root=PATH, data_name=DNAME):
    """requires torchvision import, outputs list of torch tensors (images) and list of their descriptions"""
    data = get_data(json_object_path=os.path.join(root, data_name))
    images, description = zip(*[(read_image(os.path.join(root, local_path)), data) for local_path, data in data.items()])
    print(f'{len(images)} images and their descriptions have been loaded successfully')
    return images, description


id_to_class = [Circle, Rhombus, Rectangle, Triangle, Polygon]
# have to instantiate T_T
id_to_cname = [c((100, 100), (50, 50)).shape for c in id_to_class]
cname_to_id = {cn: i for i, cn in enumerate(id_to_cname)}


class FiguresDataset(VisionDataset):
    def __init__(self, root=PATH, transforms=None):
        super().__init__(root)
        self.images, self.descriptions_ = load_dataset()
        assert len(self.images) == len(self.descriptions_), 'wrong dataset generation, please retry'
        self.transforms = transforms
        new_descriptions = []
        for desc in self.descriptions_:
            labels_list, bbox_list = [],[]
            for fig in desc:
                # [['Rhombus', [16, 32], [60.104076400856535, 60.104076400856535]], ['Hexagon', [20, 190], [51.8, 51.8]],...]
                # lay out bbox as (xmin,ymin,xmax,ymax)
                bbox = fig[1] + [fig[1][i] + fig[2][i] for i in range(2)]
                bbox_list.append(bbox)
                labels_list.append(cname_to_id[fig[0]])
            new_descriptions.append((labels_list, bbox_list))
        self.descriptions = new_descriptions

    def __getitem__(self, idx):
        return self.images[idx], self.descriptions[idx]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    # box1 = BBox((100, 100), (40, 10))
    # print(box1)
    # box2 = BBox((70, 100), (20, 10))
    # print(box2)
    # print(box1.get_iou(box2))
    # print(box1+box2)

    # print(Triangle((100, 100), (50, 50)))
    # print(Rhombus((100, 100), (20, 70)))
    # print(Polygon((100, 100), (50, 50), ratio=0.7, nv=6, angle=44))
    # print(Circle((100, 100), (30, 50), ratio=0.7))
    # print(Rectangle((120, 120), (80, 90), ratio=0.9, angle=20, sqrnss=0.2))

    # allocate Ox, Oy projections of possible rectangles
    # x, y = get_centers(), get_centers()
    # # generate product of all xs, ys (all possible combinations without replacement)
    # centers_xy, half_sizes_wh = list(product(x[0], y[0])), list(product(x[1], y[1]))
    # draw_bounds(zip(centers_xy, half_sizes_wh))

    # choice = rc_parts(centers_xy, half_sizes_wh)
    # print(f'{len(centers_xy)} rectangles (and their shapes) in total, chosen {len(choice)}')

    # get allowed rectangles
    # draw_bounds(choice)
    # result = draw_shapes(choice)
    # result[0].show()

    d = generate(n=10000, root=PATH, folder_name=FNAME, data_name=DNAME, store=False)
    tile(d, 1000).show()