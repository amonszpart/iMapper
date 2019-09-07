import argparse
import sys
import os
import glob
import numpy as np
import copy
from scipy.ndimage.filters import gaussian_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from imapper.logic.scenelet import Scenelet
from imapper.logic.skeleton import Skeleton
from imapper.logic.joints import Joint
from imapper.pose.span import Span


class Span2(Span):
    """Class to record pose-pose distance."""
    
    def __init__(self, start, end, value, actor_id=0):
        super(Span2, self).__init__(
            start=start, end=end, actor_id=actor_id,
            strict=False, charness=None, charness_unw=None)
        self._value = value
        
    @property
    def value(self):
        return self._value

    def __repr__(self):
        return 'Span2({:d}..{:d}, d={:.3f})' \
            .format(self.start, self.end, self.value)


def smooth(data):
    """Smoothes a list.
    Args:
        data (List[Tuple[str, float]]): 
            Entry pairs to smooth. 
    Returns:
        (List[Tuple[str, float]]): Smoothed entries.
    """
    
    data = sorted(data)
    keys, vals, times = zip(*data)
    vals = gaussian_filter(vals, sigma=1.5)
    data = list(zip(keys, vals, times))
    print('post: {}'.format(data))
    return data


def main(argv):
    parser = argparse.ArgumentParser(
        "Filter initial path based on distance to full fit")
    parser.add_argument('skel', help="Skeleton file to filter", type=str)
    parser.add_argument('--threshold', help='Distance threshold. Default: 0.4',
                        type=float, default=0.4)
    args = parser.parse_args(argv)
    lower_body = [Joint.LKNE, Joint.RKNE, Joint.LANK, Joint.RANK,
                  Joint.LHIP, Joint.RHIP]
    
    print(args.skel)
    p_root = os.path.dirname(args.skel)
    p_fit = os.path.join(p_root, 'opt1')
    assert os.path.isdir(p_fit), p_fit
    query = Scenelet.load(args.skel)
    out = Skeleton()
    
    data = []
    
    x = []
    y = []
    y2 = []
    for d_ in sorted(os.listdir(p_fit)):
        d = os.path.join(p_fit, d_)
        pattern = os.path.join(d, 'skel_*.json')
        for f in sorted(glob.iglob(pattern)):
            print(f)
            assert '00' in f, f
            sclt = Scenelet.load(f)
            frames = sclt.skeleton.get_frames()
            mid_frame = frames[len(frames) // 2]
            time = sclt.skeleton.get_time(mid_frame)
            q_frame_id = query.skeleton.find_time(time)
            q_time = query.skeleton.get_time(q_frame_id)
            print(time, q_time, f)
            q_pose = query.skeleton.get_pose(q_frame_id)
            pose = sclt.skeleton.get_pose(mid_frame)
            pose[[0, 2]] -= (
                    pose[:, Joint.PELV:Joint.PELV + 1]
                    - q_pose[:, Joint.PELV:Joint.PELV + 1])[[0, 2]]
            diff = np.mean(
                np.linalg.norm(q_pose[:, lower_body] - pose[:, lower_body],
                               axis=0))
            print(q_frame_id, time, diff)
            y.append(diff)
            x.append(q_frame_id)
            data.append((q_frame_id, diff, time))
            
            if query.skeleton.has_pose(q_frame_id - 1):
                tmp_pose = copy.deepcopy(q_pose)
                tmp_pose -= tmp_pose[:,
                            Joint.PELV:Joint.PELV + 1] - query.skeleton.get_pose(
                    q_frame_id - 1)[:, Joint.PELV:Joint.PELV + 1]
                y2.append(np.mean(np.linalg.norm(
                    pose[:, lower_body] - tmp_pose[:, lower_body], axis=0)))
            else:
                y2.append(0.)
            
            out.set_pose(frame_id=q_frame_id, time=q_time, pose=pose)
            break
    
    data = smooth(data)
    plt.plot(x, y, 'x--', label='Distance to best Kinect fit\'s center frame')
    plt.plot(x, y2, 'o--', label='Distance to prev pose')
    plt.plot([d[0] for d in data], [d[1] for d in data], 'o--',
             label='Smoothed')
    plt.xlabel('Time (s)')
    plt.ylabel('Sum local squared distance')
    plt.legend()
    plt.savefig(os.path.join(p_root, 'tmp.pdf'))
    Scenelet(skeleton=out).save(os.path.join(p_root, 'skel_tmp.json'))
    
    above = []
    prev_frame_id = None
    for frame_id, dist, time in data:
        # assert prev_frame_id is None or frame_id != prev_frame_id, \
        #     'No: {}'.format(frame_id)
        if dist > args.threshold:
            above.append(Span2(start=frame_id, end=frame_id, value=dist,
                               time=time))
        prev_frame_id = frame_id
        
    spans = [copy.deepcopy(above[0])]
    it = iter(above)
    next(it)
    prev_frame_id = above[0].start
    for span2 in it:
        frame_id = span2.start
        if prev_frame_id + 1 < frame_id:
            # span = spans[-1]
            # spans[-1] = span[0], prev_frame_id, span[2]
            spans[-1].end = prev_frame_id
            spans.append(
                Span2(start=frame_id, end=frame_id, time=None,
                      value=span2.value))
        else:
            print(prev_frame_id, frame_id)
        prev_frame_id = frame_id
    spans[-1].end = prev_frame_id
    print("Need replacement: {}".format(above))
    print("Need replacement2: {}".format(spans))


if __name__ == "__main__":
    main(sys.argv[1:])
