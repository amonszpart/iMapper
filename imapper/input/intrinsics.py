"""
Duygu's intrinsics: Canon G15
ExifTool:
Related Image Width             : 1920
Related Image Height            : 1080
Focal length: 6.1 mm
https://www.dpreview.com/products/canon/compacts/canon_g15/specifications
Sensor size: 7.44 x 5.58 mm

Focal length in pixels = 
    (image width in pixels) * (focal length in mm) / (CCD width in mm)
fx = 1920 * 6.1 / 7.44 = 1574.193548387
fy = 1080 * 6.1 / 5.58 = 2098.924731183

------------------------------------------------------------------------

Duygu's intrinsics: Canon T6
ExifTool:
Related Image Width             : 1920
Related Image Height            : 1080
Focal length: 18 mm
http://www.imaging-resource.com/PRODS/canon-t6/canon-t6A.HTM
Sensor size: 22.3 x 14.9 mm

Focal length in pixels =
    (image width in pixels) * (focal length in mm) / (CCD width in mm)
fx = 1920 * 18 / 22.3 = 1549.775784753
fy = 1080 * 18 / 14.9 = 1304.697986577 

------------------------------------------------------------------------

Duygu's intrinsics: Sony A7R
ExifTool:
Related Image Width             : 1920
Related Image Height            : 1080
Focal length: 18 mm
http://www.imaging-resource.com/PRODS/sony-a7r/sony-a7rA.HTM
Sensor size: 35.9mm x 24.0mm

Focal length in pixels =
    (image width in pixels) * (focal length in mm) / (CCD width in mm)
fx = 3840 * 35 / 35.9 = 3743.732590529
fy = 2160 * 35 / 24.0 = 3150

------------------------------------------------------------------------

ExifTool:
Related Image Width             : 1920
Related Image Height            : 1080
Focal length: 18 mm
http://www.imaging-resource.com/PRODS/sony-a7r/sony-a7rA.HTM
Sensor size: 35.9mm x 24.0mm

Focal length in pixels =
    (image width in pixels) * (focal length in mm) / (CCD width in mm)
fx = 3840 * 35 / 35.9 = 3743.732590529
fy = 2160 * 35 / 24.0 = 3150

------------------------------------------------------------------------

S8:
Resolution:
Still: 4032x3024
Video: 3840x2160
Focal length: 4.25mm
Other stuff: f/1.7, 26mm, 1/2.55", 1.4 µm

------------------------------------------------------------------------
"""
import numpy as np
from imapper.util.stealth_logging import lg


def intrinsics_matrix(scaled_height, shape_orig, camera_name):
    """
    Estimate intrinsic matrix for a given camera
    :param shape_orig: height x width of original image (e.g. [1080, 1920]
    :param scaled_height: pose.config.INPUT_SIZE from Denis' config.py (368)
    
    """
    K = np.eye(3, dtype='f4')
    width = np.float32(shape_orig[1] * scaled_height / shape_orig[0])
    height = np.float32(scaled_height)
    K[0, 2] = np.float32(width / 2.)
    K[1, 2] = np.float32(height / 2.)
    if camera_name.lower() in {'s6', 'aron', 'galaxy s6'}:
        # calibration happened for 1280 x 720
        K[0, 0] = np.float32((width / 1280.) * 1086.)
        K[1, 1] = np.float32((height / 720.) * 1085.)
    elif camera_name.lower() in {'g15', 'canon g15'}:
        # see above
        K[0, 0] = np.float32(width / 1920 * 1574.193548)
        K[1, 1] = np.float32(height / 1080 * 2098.924731)
    elif camera_name.lower() in {'t6', 'canon t6'}:
        # see above
        K[0, 0] = np.float32(width / 1920 * 1549.775784753)
        K[1, 1] = np.float32(height / 1080 * 1304.697986577)
    elif camera_name.lower() in {'a7r', 'sony a7r'}:
        # see above
        K[0, 0] = np.float32(width / 3840. * 3743.732590529)
        K[1, 1] = np.float32(height / 2160. * 3150.)
    elif camera_name.lower() in {'s8'}:
        lg.warning("TODO: calibrate S8")
        K[0, 0] = np.float32((width / 1280.) * 1086.)
        K[1, 1] = np.float32((height / 720.) * 1085.)
    else:
        raise RuntimeError("Unknown camera name: %s" % camera_name)

    return K
