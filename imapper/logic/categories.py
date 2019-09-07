CATEGORIES = {
    'table': 0,
    'chair': 1,
    'couch': 2,
    'shelf': 3,
    'bed': 4,
    'whiteboard': 5,
    'monitor': 6,
    'tv': 7,
    'plant': 8
}

def category_list():

    cat_list = [None] * (max(CATEGORIES.values())+1)
    for cat_name, cat_ind in CATEGORIES.items():
        cat_list[cat_ind] = cat_name

    return cat_list

IMG_DEPTH_PER_CAT = 16
"""How many angular slices for each category"""


class TensorDesc(object):
    def __init__(self, start, depth):
        assert start >= 0, "Need non-negative start: %d" % start
        assert depth > 0, "Need positive depth: %d" % depth
        self.start = start
        self.depth = depth

    @property
    def end(self):
        return self.start + self.depth

    @property
    def idx(self):
        return self.start // IMG_DEPTH_PER_CAT

    def __repr__(self):
        return "<Tensordesc(start:%d, depth:%d, end:%d)>" \
               % (self.start, self.depth, self.end)


CATEGORIES_IN = {
    'table': TensorDesc(CATEGORIES['table'] * IMG_DEPTH_PER_CAT, IMG_DEPTH_PER_CAT),
    'chair': TensorDesc(CATEGORIES['chair'] * IMG_DEPTH_PER_CAT, IMG_DEPTH_PER_CAT),
    'couch': TensorDesc(CATEGORIES['couch'] * IMG_DEPTH_PER_CAT, IMG_DEPTH_PER_CAT)
    # 'shelf': TensorDesc(CATEGORIES['shelf'] * IMG_DEPTH_PER_CAT, IMG_DEPTH_PER_CAT)
}
# NOTE, if shelf is back, couch below needs to change to shelf!!!
CATEGORIES_IN['pelvis'] = TensorDesc(CATEGORIES_IN['couch'].end, 1)
CATEGORIES_IN['skel_mask'] = TensorDesc(CATEGORIES_IN['pelvis'].end, 1)
CATEGORIES_IN['skel_fw'] = TensorDesc(CATEGORIES_IN['skel_mask'].end, 2)
CATEGORIES_IN['skel_y'] = TensorDesc(CATEGORIES_IN['skel_fw'].end, 2)

COLORS_CATEGORIES = {
    # 'chair': (252., 141., 98.),
    'table': (166., 216., 84.),
    # 'couch': (102., 194., 165.),
    'sittable': (102., 141., 135.)
}

TRANSLATIONS_CATEGORIES = {
    'sofa': 'couch',
    'stool': 'chair',
    'coffeetable': 'table',
    'bookshelf': 'shelf',
    'nightstand': 'shelf'
}

CATEGORIES_DOMINANT = frozenset(('table'))
"""Categories that can overwrite a top-view pixel (this is a hack z-test)"""

# CATEGORIES_OUT = {k: v for k, v in CATEGORIES.items()}
CATEGORIES_OUT = {'table': 0}
CATEGORIES_OUT['sittable'] = len(CATEGORIES_OUT)
CATEGORIES_OUT['empty'] = len(CATEGORIES_OUT)
