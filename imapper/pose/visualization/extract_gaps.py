import argparse
import copy
import glob
import json
import os
import sys
from itertools import groupby, count, chain
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from imapper.logic.joints import Joint
from imapper.logic.scenelet import Scenelet
from imapper.pose.filter_full_fit import Span2
from imapper.pose.span import Span

import numpy as np
from scipy.ndimage.filters import gaussian_filter


_lower_body = [Joint.LKNE, Joint.RKNE,
               Joint.LANK, Joint.RANK,
               Joint.LHIP, Joint.RHIP]


def _check_exists(l):
    if os.path.exists(l):
        return l
    else:
        l = os.path.join(os.path.dirname(__file__), l)
        if os.path.exists(l):
            return l
    
    raise RuntimeError("Path does not exist: \"%s\"" % l)


def get_pose_diff(pose_q, pose_m):
    """Estimates distance with aligned pelvises in 2D.
    
    Args:
        pose (np.ndarray): (3, 16) Pose to compare.
        q_pose (np.ndarray): (3, 16) Pose to compare to.
        
    Returns:
        (float): Pose distance
    """
    
    # TODO: Project onto ground instead of assume y-down
    pose_m[[0, 2]] -= (pose_m[:, Joint.PELV:Joint.PELV + 1]
                       - pose_q[:, Joint.PELV:Joint.PELV + 1])[[0, 2]]
    diff = np.mean(np.linalg.norm(pose_q[:, _lower_body]
                                  - pose_m[:, _lower_body], axis=0))
    
    return diff


def group_spans(above):
    """Groups consecutive entries to spans.
    
    Args:
        above (List[Span2]): Sorted spans with length == 1, having
        distance above the distance threshold.
        
    Returns:
         (List[Span2]): Spans with length >=1, the min distance between
         two consecutive spans is now > 1.
         Distance := second.start - first.end.
         
    """
    if not len(above):
        return []
    
    assert all(a.end <= b.start for a, b in zip(above[:-1], above[1:])), \
        'Assumed sorted spans: {}'.format(above)
    spans = [copy.deepcopy(above[0])]
    it = iter(above)
    next(it)
    prev_frame_id = above[0].start
    for span2 in it:
        frame_id = span2.start
        if prev_frame_id + 1 < frame_id:
            spans[-1].end = prev_frame_id
            spans.append(Span2(start=frame_id, end=frame_id,
                               value=span2.value))
        prev_frame_id = frame_id
    spans[-1].end = prev_frame_id
    
    return spans


def get_delta_frame_id(span0, span1):
    """Computes the minimum distance between two non-overlapping spans.
    
    Args:
         span0 (Span): First span.
         span1 (Span): Second span.
    """
    if span0.overlaps(span1):
        assert False, (span0, span1)
        return 0
    
    if span0.end < span1.start:
        return span1.start - span0.end
    else:
        return span0.start - span1.end


def filter_spans_to_replace(spans, chosen, c_threshold):
    """Estimates remaining spans that are not covered in chosen.
    
    Args:
        spans (List[Span2]):
            Spans to replace because of distance (will be used without
            objects).
        chosen (List[Span]):
            Spans chosen for candidate generation (will be used with
            objects).
        c_threshold (float):
            Charness threshold.
        
    Returns:
        (List[Span2]): Spans to add to gap_command.sh with the
        --remove-objects flag.
    """
    # keep track of activated actor ids (we'll keep one each)
    actor_ids = set()
    
    # sort in decreasing characteristicness order
    chosen_srtd = sorted(chosen, key=lambda _c: _c.charness, reverse=True)
    
    # filter through all spans
    for c in chosen_srtd:
        
        # if not characteristic enough and actor already covered
        if c.actor_id in actor_ids and c.charness < c_threshold:
            c.active = False
            continue
            
        # we'll need at least one scenelet for each actor
        actor_ids.add(c.actor_id)
        
        # find closest active Span
        closest = min(
            (c2 for c2 in chosen if c2 != c and c2.active),
            key=lambda _c: get_delta_frame_id(c, _c))
        
        # time difference to closest active Span
        dt = get_delta_frame_id(c, closest)
        
        # remove, if has a more characteristic neighbour that's active
        # TODO: this might be unnecessary, could be swapped for Span2
        # immediately
        if dt == 1 and closest.active and closest.charness > c.charness:
            c.active = False
            
    # re-activate inactive ones above threshold without objects (Span2)
    # for i in range(len(chosen_srtd)):
    #     c = chosen_srtd[i]
    #     if c.active or c.charness < c_threshold:
    #         continue
    #
    #     # Span2 means without objects
    #     chosen_srtd[i] = Span2(start=c.start, end=c.end, value=99.,
    #                            actor_id=c.actor_id)
        
    # find uncovered parts
    chosen_set = frozenset(
        f for s in chosen for f in range(s.start, s.end + 1) if s.active)
    # gather output spans
    o_spans = []
    for span in spans:
        frames = frozenset(f for f in range(span.start, span.end + 1))
        remainder = sorted(frames.difference(chosen_set))
        if not len(remainder):
            continue
        
        for _, it in groupby(remainder, lambda n, k=count(): n - next(k)):
            l_it = list(it)
            start, end = l_it[0], l_it[-1]
            if start == end:
                end = end + 1
            o_spans.append(Span2(start=start, end=end, value=span.value,
                                 actor_id=span.actor_id))
    print('o_spans: {}'.format(o_spans))
    
    return chosen, o_spans


def get_pose_distance(query3d, path_match, gap):
    skel_q = query3d.skeleton
    mid_frame = gap[0] + (gap[1] - gap[0]) // 2
    frame_q = min((frame_id for frame_id in skel_q.get_frames()),
                  key=lambda frame_id: abs(frame_id - mid_frame))
    time = skel_q.get_time(frame_id=frame_q)
    print('Closest to {} is {} with time {}'.format(mid_frame, frame_q, time))
    match = Scenelet.load(path_match)
    skel_m = match.skeleton
    frame_m = skel_m.find_time(time=time)
    time_m = skel_m.get_time(frame_id=frame_m)
    print('Closest match is {} with time {}'.format(frame_m, time_m))
    
    diffs = []
    for frame_id in range(frame_q - 1, frame_q + 2):
        if not skel_q.has_pose(frame_id):
            print('Skipping frame_id {} in query because missing'
                  .format(frame_id))
            continue
        _time_q = skel_q.get_time(frame_id=frame_id)
        _frame_m = skel_m.find_time(time=_time_q)
        _time_m = skel_m.get_time(frame_id=_frame_m)
        if abs(_time_m - _time_q) > 1.:
            print('Skipping matched time of {} because too far from {}'
                  .format(_time_m, _time_q))
            continue
        pose_q = skel_q.get_pose(frame_id=frame_id)
        pose_m = skel_m.get_pose(frame_id=_frame_m)
        diff = get_pose_diff(pose_q=pose_q, pose_m=pose_m)
        print('Diff: {}'.format(diff))
        diffs.append(diff)
    
    # return np.mean(diffs), len(diffs) > 0
    return np.max(diffs), len(diffs) > 0


def find_surrounding(span, spans):
    """Finds the span in spans that best surrounds span.
   
    Args:
        span (Span): Span to surround.
        spans (List[Span]): Surrounding candidates.
        
    Returns:
        (Span): Surrounding span.
    """
    mid = (span.start + span.end) // 2
    return max((s for s in spans if s.start < mid < s.end),
               key=lambda _s: min(_s.end - mid, mid - _s.start))


def construct_command(name_query, span, tc, surr):
    comm = '# ' if not span.active else ''
    remove_objects = ' --remove-objects' \
        if span.active and isinstance(span, Span2) \
        else ''
    # if not span.active:
    #     cmd = "# %s" % cmd
    # elif isinstance(span, Span2):
    #     cmd = '{} --remove-objects'.format(cmd)
    
    path_root = '/media/data/amonszpa/stealth/shared/video_recordings'
    cmd = "{comm:s}D=\"{root:s}/{scene:s}/opt2b/{t0:03d}_{t1:03d}\";\n" \
          "{comm:s}if [ ! -d ${{D}} ]; then\n" \
          "{comm:s}\tpython3 stealth/pose/opt_consistent.py -silent " \
          "--wp 1 --ws 0.05 --wi 1. --wo 1. " \
          "-w-occlusion -w-static-occlusion --maxiter 15 " \
          "--nomocap -v " \
          "{root:s}/{scene:s}/skel_{scene:s}_unannot.json " \
          "independent " \
          "-s /media/data/amonszpa/stealth/shared" \
          "/pigraph_scenelets__linterval_squarehist_large_radiusx2_" \
          "smoothed_withmocap_ct_full_sampling " \
          "--gap {t0:03d} {t1:03d} --batch-size 10 --dest-dir opt2 " \
          "--candidates opt1/{s0:03d}_{s1:03d} --n-candidates 200 " \
          "-tc {tc:.2f}{remove_objects:s}\n" \
          "{comm:s}fi" \
        .format(scene=name_query, t0=span.start, t1=span.end,
                tc=tc, s0=surr.start, s1=surr.end,
                root=path_root, comm=comm,
                remove_objects=remove_objects)
    return cmd


class LoggedSequence(object):
    def __init__(self):
        self._data = []
    
    def add_point(self, value, time):
        assert isinstance(time, (int, float, np.float32, np.float64)), \
            print("Expected time, not {} ({})".format(time, type(time)))
        self._data.append([value, time])
        
    def plot(self, path_root):
        values, times = zip(*self._data)
        print('values: {}\ntimes: {}'.format(values, times))
        f = plt.figure()
        ax = f.add_subplot(111)

        ax.plot(times, values, 'x--',
                label='Distance to best Kinect fit\'s center frame')
        plt.xlabel('Time (s)')
        plt.ylabel('Sum local squared distance')
        plt.legend()
        plt.savefig(os.path.join(path_root, 'tmp.pdf'))


def main(argv):
    pjoin = os.path.join
    
    parser = argparse.ArgumentParser("")
    parser.add_argument('d', type=_check_exists,
                        help="Input directory")
    parser.add_argument(
        '-o', '--opt-folder',
        help="Which optimization output to process. Default: opt1",
        default='opt1')
    parser.add_argument(
        '-limit', type=int, help="How many scenelets to aggregate.",
        default=3
    )
    parser.add_argument('--c-threshold',
                        help='Distance threshold. Default: 0.3',
                        type=float, default=0.3)
    parser.add_argument('--d-threshold',
                        help='Distance threshold. Default: 0.4',
                        type=float, default=0.4)
    args = parser.parse_args(argv)
    print("Working with %s" % args.d)
    d = pjoin(args.d, args.opt_folder)
    
    name_query = d.split(os.sep)[-2]
    query3d = Scenelet.load(
        os.path.join(args.d,
                     'skel_{:s}_unannot.json'.format(name_query)))
    n_actors = query3d.skeleton.n_actors  # type: int
   
    log = LoggedSequence()
    elements = [[], [], []]
    spans = []
    above = []
    for p in sorted(os.listdir(d)):
        d_time = pjoin(d, p)
        if not os.path.isdir(d_time) or 'bak' in p:
            continue
        parts = p.split('_')
        start = int(parts[0])
        end = int(parts[-1])
        
        sum_charness = 0.
        sum_weight = 0.
        sum_charness_unw = 0.
        sum_weight_unw = 0
        diff_succ, diff = False, 100.
        for f in glob.iglob("%s/skel_%s_*.json" % (d_time, name_query)):
            rank = int(os.path.splitext(f)[0].rpartition('_')[-1])
            if rank >= args.limit:
                continue
            # print(f)
            data = json.load(open(f, 'r'))
            charness = data['charness']
            weight = max(0., .1 - data['score_fit'])
            sum_charness += weight * charness
            sum_weight += weight
            sum_charness_unw += charness
            elements[sum_weight_unw].append(charness)
            sum_weight_unw += 1
            
            if rank == 0:
                diff, diff_succ = get_pose_distance(
                    query3d=query3d, path_match=f, gap=(start, end))
        
        if sum_weight > 0.:
            sum_charness /= sum_weight
        if sum_weight_unw > 0.:
            sum_charness_unw /= sum_weight_unw
        frame_id = (start + end) // 2
        actor_id = query3d.skeleton.get_actor_id(frame_id=frame_id) \
            if n_actors > 1 else 0
        
        spans.append(Span(start, end, sum_charness, sum_charness_unw,
                          actor_id=actor_id))
        
        # check for pose replacement
        if diff_succ and diff > args.d_threshold:
            # time = query3d.skeleton.get_time(frame_id=frame_id)
            above.append(Span2(start=frame_id, end=frame_id, value=diff))

        if diff_succ:
            log.add_point(value=diff, time=frame_id)
    
    for actor_id in range(n_actors):
        cs, span_ids = zip(*[(span.charness, span_id)
                             for span_id, span in enumerate(spans)
                             if span.actor_id == actor_id])
        cs2 = gaussian_filter(cs, sigma=2.5).tolist()
        
        for smoothed, span_id in zip(cs2, span_ids):
            spans[span_id].smoothed_charness = smoothed
    
    plt.figure()
    plt.plot(cs, label="orig")
    plt.plot([span.smoothed_charness for span in spans], label="smoothed")
    # c0 = [span.charness_unw for span in spans]
    # plt.plot(c0, 'o-', label="unweighted")
    # for sigma in [2.5]:
    #     cs2 = gaussian_filter(cs, sigma=sigma)
    #     plt.plot(cs2, '+-', label="%f" % sigma)
    
    # for i, elem in enumerate(elements):
    #     plt.plot(elem, 'x-', label="cand%d" % i)
    plt.legend()
    # plt.show()
    p_graph = os.path.join(args.d, 'charness2.svg')
    plt.savefig(p_graph)
    
    log.plot(path_root=args.d)
    
    spans = sorted(spans, key=lambda s: s.smoothed_charness, reverse=True)
    
    chosen = [spans[0]]
    for span_id in range(1, len(spans)):
        span = spans[span_id]
        overlap = next((c for c in chosen if span.overlaps(c)), None)
        if overlap is not None:
            # print("Skipping %s because overlaps %s" % (span, overlap))
            continue
        elif span.start == query3d.skeleton.get_frames()[0] \
                or span.end == query3d.skeleton.get_frames()[-1]:
            print("Skipping %s because first/last frame." % span)
            continue
        else:
            chosen.append(span)
    
    # Spans to replace, because of quality, if not replaced already
    spans_r = group_spans(above=above)
    chosen, spans_r_filtered = filter_spans_to_replace(
        spans=spans_r, chosen=chosen, c_threshold=args.c_threshold)
    print('Filtered: {}\nchosen: {}'.format(spans_r_filtered, chosen))
    # keep at least one characteristic pose for each actor
    # actors_output = set()
    p_cmd = os.path.join(args.d, 'gap_command_new.sh')
    with open(p_cmd, 'w') as f:
        for actor_id in range(n_actors):
            f.write('######\n# Actor {:d}\n######\n\n'.format(actor_id))
            for span in chain(chosen, spans_r_filtered):
                if not span.actor_id == actor_id:
                    continue
                surr = find_surrounding(span, spans)
                cmd = construct_command(name_query=name_query, span=span,
                                        tc=args.c_threshold, surr=surr)
                # if not span.active:
                #     cmd = "# %s" % cmd
                # elif isinstance(span, Span2):
                #     cmd = '{} --remove-objects'.format(cmd)
                    
                if isinstance(span.charness, float):
                    f.write("# charness: %g\n" % span.charness)
                
                # Comment out if too low
                # if span.charness < args.c_threshold \
                #         and actor_id in actors_output:
                # actors_output.add(actor_id)
                f.write('{}\n\n'.format(cmd))
            f.write('\n\n')
        # with open(pjoin(d_time, 'avg_charness.json')) as fch:
        #     data = json.load(fch)
        #     charness = data['avg_charness']
        #     print(charness)


if __name__ == '__main__':
    main(sys.argv[1:])
