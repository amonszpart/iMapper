import sys
import platform

from imapper.util.stealth_logging import lg

if 'matplotlib' not in sys.modules:
    import matplotlib as matplotlib
    matplotlib.use('Agg')
    matplotlib.rc('font', family='DejaVu Sans')
else:
    matplotlib = sys.modules['matplotlib']
if 'plt' not in sys.modules:
    import matplotlib.pyplot as plt



def show_image(ax, image, room, vmin=None, vmax=None):
    assert len(image.shape) == 2, \
        "Need a 2D image: %s" % repr(image.shape)

    ax.imshow(
        image,
        cmap='jet', interpolation='nearest', aspect='equal',
        extent=(room[2, 0], room[2, 1], -room[0, 1], -room[0, 0]),
        vmin=vmin, vmax=vmax)


def show_points(ax, points, values):
    assert len(points.shape) == 2, \
        "Need a list of 2D points: %s" % repr(points.shape)
    if points.shape[0] == 2:
        points = points.T
    assert points.shape[1] == 2, \
        "Need a list of 2D points: %s" % repr(points.shape)
    ax.scatter(points[:, 1], -points[:, 0], c=values)

try:
    from descartes import PolygonPatch
    import shapely.geometry as geom

    def show_polygon(ax, polygon, facecolor=None, alpha=1.):
        ax.add_artist(PolygonPatch(geom.asPolygon(polygon), facecolor=facecolor,
                                   alpha=alpha))
except ImportError:
    pass
