import numpy
import os
import json

import random
import statistics
from math import pi, sin, cos, sqrt, atan
from itertools import accumulate, product
from PIL import Image, ImageDraw

RSEED = 42
PATH = './'

SIZE = 256
THR = 25
PARTS = 5
MARGIN = 3


# random.seed(RSEED)
# numpy.random.seed(RSEED)
# TODO 1: generate random axis split (non-overlapping, up to 5)
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

def get_centers(n=SIZE, k=PARTS, l_bound=THR):
    """convenient function that generates by default """
    spacers = rand_split(n, k, l_bound)
    delimiters = [0] + list(accumulate(spacers))[:-1] + [256]
    centers = [round(p + (f - p) / 2) for f, p in zip(delimiters[1:], delimiters[:-1])]
    half_sizes = [s / 2 for s in spacers]
    print(f'We have {len(spacers)} intervals allocated as {delimiters}, their centers are {centers}')
    return centers, half_sizes


# TODO 2: set up classes


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


# box1 = BBox((100, 100), (40, 10))
# print(box1)
# # print(box.ve_min, box.ve_max)
# box2 = BBox((70, 100), (20, 10))
# print(box2)

# print(box1.get_iou(box2))
# print(box1+box2)


class Rectangle(Figure):
    def __init__(self, center, half_size, rotation_angle_r=0):
        super().__init__(center, half_size)
        self.shape = self.__class__.__name__
        self.rotation = rotation_angle_r
        # available space
        self.bbox = BBox(center, half_size)
        # crop to square w/ margin
        self.bbox_ = BBox(center, (min(half_size), min(half_size)))
        self.wh = self.bbox_.wh
        # max possible inscribed circle and implying rotation
        self.max_radius = (min(half_size) - MARGIN) / sqrt(2)
        # global coordinates of farthest vertices of an inner square (to be shown)
        self.ve_min = self.x - self.max_radius, self.y - self.max_radius
        self.ve_max = self.x + self.max_radius, self.y + self.max_radius
        self.ve = self.ve_min, self.ve_max
        self.ve_loc = ((-self.max_radius, -self.max_radius), (self.max_radius, self.max_radius))
        # polar form of local coordinates w/corrections for angles more than [-pi, pi]
        self.ve_loc_polar = list((sqrt(x**2 + y**2), pi+atan(y/x)) if y < 0 else (sqrt(x**2 + y**2), atan(y/x)) for (x, y) in self.ve_loc)

    def rotate(self, angle_r):
        self.ve_loc_polar = list((r, phi + angle_r) for r, phi in self.ve_loc_polar)
        # overwrite vertices
        self.ve = sorted(list((self.x + r*cos(phi), self.y + r*sin(phi)) for r, phi in self.ve_loc_polar), key=lambda t:t[0])

    def __repr__(self):
        image = Image.new(mode='RGB', size=(SIZE, SIZE), color='white')
        canvas = ImageDraw.Draw(image)
        canvas.rectangle(self.ve, fill='black')
        print(self.ve)
        self.rotate(pi/3)
        print(self.ve)
        # canvas.rectangle(self.ve, fill='yellow')
        canvas.rectangle(self.bbox.ve, outline='green')
        canvas.rectangle(self.bbox_.ve, outline='red')
        canvas.ellipse(self.bbox_.ve, outline='red')
        canvas.point((self.x, self.y), 'red')
        canvas.point(self.ve[0], 'red')
        canvas.point(self.ve[1], 'red')
        # canvas.point(self.ve[0], 'blue')
        # canvas.point(self.ve[1], 'blue')
        image.show()
        return f'{self.shape} with vertices {self.ve} bounded by {self.bbox_.ve}'

# w = Rectangle((100, 100), (40, 70))
# w.rotate(pi/5)
# print(w)

class Triangle(Figure):
    def __init__(self, center, half_size, var_threshold=0.3):
        super().__init__(center, half_size)
        self.shape = self.__class__.__name__
        self.bbox_ = BBox(center, half_size)
        shift_x = list(range(-half_size[0] + MARGIN, half_size[0] - MARGIN))
        shift_y = list(range(-half_size[1] + MARGIN, half_size[1] - MARGIN))
        random.shuffle(shift_x)
        random.shuffle(shift_y)
        # prevent slim triangles as it's hard to distinguish those
        var_x, var_y = 0, 0
        while var_x * var_y < var_threshold * (half_size[0] * half_size[1]):
            random.shuffle(shift_x)
            random.shuffle(shift_y)
            var_x, var_y = statistics.stdev(shift_x[:3]), statistics.stdev(shift_y[:3])
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


class Rhombus(Figure):
    def __init__(self, center, half_size, s=0.5):
        """larger s -- thinner rhombus"""
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

        lim_xs = ((self.bbox.ve_min[0] + MARGIN // s), (self.bbox.ve_max[0] - MARGIN // s))
        # let's randomly generate max of side xs
        # print(self.bbox.x + MARGIN // s, self.bbox.ve_max[0] - MARGIN)
        side_x = random.randint(self.bbox.x + MARGIN // s, self.bbox.ve_max[0] - MARGIN)
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
        self.ve = [(self.x + self.radius * cos(a), self.y + self.radius * sin(a)) for a in self.angles_r]

    def __repr__(self):
        image = Image.new(mode='RGB', size=(SIZE, SIZE), color='white')
        canvas = ImageDraw.Draw(image)
        canvas.polygon(self.ve, fill='black')
        canvas.rectangle(self.bbox.ve, outline='green')
        canvas.rectangle(self.bbox_.ve, outline='red')
        canvas.point((self.x, self.y), fill='white')
        image.show()
        return f'{self.shape} with angles {self.angles_d} and radius {self.radius} bounded by green box ~ {self.bbox.ve}'


class Circle(Figure):
    def __init__(self, center, half_size, ratio=0.7):
        super().__init__(center, half_size)
        self.shape = self.__class__.__name__
        self.bbox_ = BBox(center, half_size)
        self.max_radius = min(half_size) - MARGIN
        self.radius = ratio * self.max_radius
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


# print(Triangle((100, 100), (50, 50)))
print(Rhombus((100, 100), (20, 70)))
# print(Polygon((100, 100), (50, 50), ratio=0.7, nv=4, angle=20))
# print(Circle((100, 100), (30, 50), ratio=0.7))
# TODO 3: choose visual framework

def rc_parts(xy_list, wh_list):
    """choose up to 5 different boxes"""
    # random choice without repetitions
    idx = list(range(len(xy_list)))
    random.shuffle(idx)
    # random quantity from 1..PARTS
    q = random.randint(1, PARTS)
    return [(xy_list[i], wh_list[i]) for i in idx[:q]]


def draw_bounds(res):
    image = Image.new(mode='RGB', size=(SIZE, SIZE), color='white')
    canvas = ImageDraw.Draw(image)
    for b in res:
        b_ = b[0], (map(lambda d: d - MARGIN, b[1]))
        box = BBox(*b_)
        canvas.rectangle(box.ve, fill='black')
    image.show()

# # allocate Ox, Oy projections of possible rectangles
# x, y = get_centers(), get_centers()
# # generate product of all xs, ys (all possible combinations without replacement)
# centers_xy, half_sizes_wh = list(product(x[0], y[0])), list(product(x[1], y[1]))
# choice = rc_parts(centers_xy, half_sizes_wh)
# print(f'{len(centers_xy)} rectangles (and their shapes) in total, chosen {len(choice)}')
# # draw_bounds(choice)
# get allowed rectangles
# print(choice)
# TODO 4: describe (id,type,x,y,w,h)
# TODO 5: inscribe 4 shapes into those rectangles, fill-in with colours
# TODO 6: serialize via json
# TODO 7: create dataset (png image <--> json description)
