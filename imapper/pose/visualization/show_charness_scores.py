import argparse
import os
import sys

if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(
      os.path.realpath(__file__)), '../../../')))

from imapper.visualization.plotting import plt
from imapper.logic.scenelet import Scenelet
from imapper.logic.skeleton import Skeleton
from imapper.util.my_argparse import argparse_check_exists
from imapper.util.json import json
from imapper.util.stealth_logging import lg
from imapper.pose.visualization.extract_gaps import main as extract_gaps

import numpy as np
from math import ceil


def parser_video(parser):
    parser.add_argument("-v", '--video', type=argparse_check_exists,
                        help="Input path")
    parser.add_argument("-d", type=str,
                        help="name of folder to plot", required=True)
    return parser


def set_or_max(dict_, key, value):
    if key in dict_:
        dict_[key] = max(dict_[key], value)
    else:
        dict_[key] = value
    return dict_


def show_folder(argv):
    # python3 stealth/pose/fit_full_video.py --show /home/amonszpa/workspace/stealth/data/video_recordings/scenelets/lobby15 opt1
    # python3 stealth/pose/visualization/show_charness_scores.py --show /media/data/amonszpa/stealth/shared/video_recordings/library1 -o opt1
    pjoin = os.path.join

    parser = argparse.ArgumentParser("Fit full video")
    parser.add_argument('--show', action='store_true')
    parser.add_argument("video", type=argparse_check_exists,
                        help="Input path")
    parser.add_argument(
      '-o', '--opt-folder',
      help="Which optimization output to process. Default: opt1",
      default='opt1')
    parser.add_argument("--window-size", type=int,
                        help="Window size in frames.", default=20)

    args = parser.parse_args(argv)
    d = os.path.join(args.video, args.opt_folder)
    assert os.path.exists(d), "does not exist: %s" % d

    # parse video path
    if args.video.endswith(os.sep):
        args.video = args.video[:-1]
    name_query = os.path.split(args.video)[-1]
    print("split: %s" % repr(os.path.split(args.video)))
    p_query = pjoin(args.video, "skel_%s_unannot.json" % name_query) \
        if os.path.isdir(args.video) else args.video
    assert p_query.endswith('.json'), "Need a skeleton file"

    # load initial video path (local poses)
    query = Scenelet.load(p_query, no_obj=True)

    frame_ids = query.skeleton.get_frames()
    centroids = Skeleton.get_resampled_centroids(start=frame_ids[0],
                                                 end=frame_ids[-1],
                                                 old_frame_ids=frame_ids,
                                                 poses=query.skeleton.poses)

    depths_times_charnesses = []
    skeleton = Skeleton()
    depths = []
    skeleton.charness_poses = {}  # this is in Scenelet incorrectly...
    skeleton.score_fit = {}  # inventing this now
    skeleton.score_reproj = {}  # inventing this now
    for p in sorted(os.listdir(d)):
        d_time = pjoin(d, p)
        if not os.path.isdir(d_time):
            continue
        p_skel = next(f for f in os.listdir(d_time)
                      if os.path.isfile(pjoin(d_time, f))
                      and f.startswith('skel') and f.endswith('json')
                      and '_00' in f)
        sclt = Scenelet.load(pjoin(d_time, p_skel))
        mn, mx = sclt.skeleton.get_frames_min_max()
        frame_id = mn + (mx - mn) // 2
        if query.skeleton.has_pose(frame_id):
            pos_3d = query.skeleton.get_centroid_3d(frame_id)
        else:
            lin_id = frame_id - frame_ids[0]
            pos_3d = centroids[lin_id, :]

        # put centroid for each joint
        skeleton.set_pose(frame_id=frame_id, pose=np.tile(pos_3d[:, None],
                                                          (1, 16)))
        with open(pjoin(d_time, 'avg_charness.json')) as fch:
            data = json.load(fch)
            set_or_max(skeleton.charness_poses, frame_id, data['avg_charness'])
            # if frame_id in skeleton.charness_poses:
            #     lg.warning("Maxing charness at frame %d" % frame_id)
            #     skeleton.charness_poses[frame_id] = max(
            #       skeleton.charness_poses[frame_id], data['avg_charness'])
            # else:
            #     skeleton.charness_poses[frame_id] = data['avg_charness']

        # fit scores
        if 'score_fit' in sclt.aux_info:
            set_or_max(skeleton.score_fit, frame_id,
                       sclt.aux_info['score_fit'])
        else:
            set_or_max(skeleton.score_fit, frame_id, 0.)

        if 'score_reproj' in sclt.aux_info:
            set_or_max(skeleton.score_reproj, frame_id,
                       sclt.aux_info['score_reproj'])
        else:
            set_or_max(skeleton.score_reproj, frame_id, 0.)

    fig = plt.figure(figsize=(16, 12), dpi=100)
    ax = fig.add_subplot(121, aspect='equal')
    X = []  # skeleton x
    Z = []  # skeleton z (depth)
    C = []  # charness
    F = []  # score_fit
    R = []  # score_reproj
    T = []  # times
    for frame_id in skeleton.get_frames():
        c = skeleton.get_joint_3d(6, frame_id=frame_id)
        X.append(c[0])
        Z.append(c[2])
        C.append(skeleton.charness_poses[frame_id])
        F.append(skeleton.score_fit[frame_id])
        R.append(skeleton.score_reproj[frame_id])
        T.append(frame_id)
    ax.plot(X, Z, 'k--')
    for frame_id in skeleton.get_frames():
        if frame_id % 5:
            continue
        c = skeleton.get_joint_3d(6, frame_id=frame_id)
        ax.annotate("%d" % frame_id, xy=(c[0], c[2]), zorder=5)
    cax = ax.scatter(X, Z, c=C, cmap='jet', zorder=5)
    fig.colorbar(cax)
    z_lim = (min(Z), max(Z))
    z_span = (z_lim[1] - z_lim[0]) // 2
    x_lim = min(X), max(X)
    x_span = (x_lim[1] - x_lim[0]) // 2
    pad = .5
    dspan = z_span - x_span
    if dspan > 0:
        ax.set_xlim(x_lim[0] - dspan - pad, x_lim[1] + dspan + pad)
        ax.set_ylim(z_lim[0] - pad, z_lim[1] + pad)
    else:
        ax.set_xlim(x_lim[0] - pad, x_lim[1] + pad)
        ax.set_ylim(z_lim[0] + dspan - pad, z_lim[1] - dspan + pad)
    ax.set_title('Fit score weighted characteristicness\ndisplayed at '
                 'interpolated initial path position')

    ax = fig.add_subplot(122)

    ax.plot(T, C, 'x--', label='max charness')
    charness_threshes = [0.4, 0.35, 0.3]
    mn_thr_charness = min(charness_threshes)
    mx_thr_charness = max(charness_threshes)
    for ct in charness_threshes:
        ax.plot([T[0], T[-1]], [ct, ct], 'r')
        ax.annotate("charness %g" % ct, xy=(T[0], ct + 0.005))

    charness_sorted = sorted([(fid, c)
                              for fid, c in skeleton.charness_poses.items()],
                             key=lambda e: e[1])

    to_show = []
    # Fitness
    divisor = 5.
    F_ = -np.log10(F) / divisor
    print(F_)
    ax.plot(T, F_, 'x--', label="-log_10(score) / %.0f" % divisor)
    mx_F_ = np.percentile(F_, 90)  # np.max(F_)
    for i, (t, f) in enumerate(zip(T, F_)):
        if f > mx_F_ or any(C[i] > ct for ct in charness_threshes):
            to_show.append(i)
            # ax.annotate("%.4f" % (F[i]), xy=(t, f), xytext=(t+4, f-0.02),
            #             arrowprops=dict(facecolor='none', shrink=0.03))
            # charness
            # ax.annotate("%.3f\n#%d" % (C[i], t), xy=(t, C[i]),
            #             xytext=(t-10, C[i]-0.02),
            #             arrowprops=dict(facecolor='none', shrink=0.03))

    windows = [] # [(t_start, t_max, t_end), ...]
    crossings = {}

    # Reproj
    R_ = -np.log10(R) / divisor
    # ax.plot(T, R_, 'x--', label="-log_10(score reproj) / %.0f" % divisor)
    mx_R_ = np.max(R_)
    is_above = [False for _ in charness_threshes]
    mx_above = []
    for i, (t, r) in enumerate(zip(T, R_)):
        # if i in to_show:
            # ax.annotate("%.4f" % (R[i]), xy=(t, r), xytext=(t-10, r+0.02),
            #             arrowprops=dict(facecolor='none', shrink=0.03))
            # ax.annotate("%d" % t, xy=(t, r - 0.01))
        if (i + 1 < len(C)) and (C[i] > C[i+1]) and (C[i] > mn_thr_charness):
            mx_above.append((C[i], t))
        for thr_i, thr in enumerate(charness_threshes):
            if (C[i] > thr) != is_above[thr_i] \
              or (C[i] > mx_thr_charness and not is_above[thr_i]):
                step = 15 * (len(charness_threshes) - thr_i) \
                    if is_above[thr_i] \
                    else -15 * thr_i

                if is_above[thr_i]:
                    if 'down' not in crossings:
                        crossings['down'] = (C[i], t)
                    # else:
                    #     assert crossings['down'][0] > C[i], (crossings['down'][0], C[i])
                else:
                    if 'up' not in crossings:
                        crossings['up'] = (C[i-1], t)
                    elif crossings['up'][0] < C[i-1]:
                        crossings['up'] = (C[i-1], t)

                # ax.annotate("%.3f\n#%d" % (C[i], t), xy=(t, C[i]),
                #             xytext=(t + step, C[i]-0.1),
                #             arrowprops=dict(facecolor='none', shrink=0.03))
                if C[i] < mn_thr_charness and is_above[thr_i]:
                    try:
                        c, t = max((e for e in mx_above),
                                   key=lambda e: e[0])
                        ax.annotate("%.3f\n#%d" % (c, t), xy=(t, c),
                                    xytext=(t + step, c+0.1),
                                    arrowprops=dict(facecolor='none', shrink=0.03))
                        mx_above = []
                        windows.append((crossings['up'][1], t,
                                        crossings['down'][1]))
                    except (KeyError, ValueError):
                        lg.warning("Can't find gap: %s, %s"
                                   % (crossings, mx_above))
                    crossings = {}

                is_above[thr_i] = C[i] > thr
                break

    for crossing in windows:
        for i, t in enumerate(crossing):
            c = skeleton.charness_poses[t]
            step = -15 + i * 10
            ax.annotate("%.3f\n#%d" % (c, t), xy=(t, c),
                        xytext=(t + step, c-0.1),
                        arrowprops=dict(facecolor='none', shrink=0.03))

    extract_gaps([args.video])

    if False:
        with open(pjoin(d, 'slots.json'), 'w') as f:
            json.dump(windows, f)

        with open(pjoin(args.video, 'opt1', 'slots.json'), 'r') as f:
            windows = json.load(f)
            for window in windows:
                t0 = window[0]
                t1 = window[2]

                # if too small, make it at least 20
                if t1 - t0 < args.window_size:
                    diff = int(ceil((args.window_size - (t1 - t0)) / 2.))
                    t0 -= diff
                    t1 += diff
                assert t1 - t0 >= args.window_size

                # which candidates to use (need a 20 window)
                span = (t1 - t0) // 2
                g0 = window[1] - span
                g1 = window[1] + span
                cmd = "python3 stealth/pose/opt_consistent.py -silent " \
                      "--wp 10 --ws 0.05 --wi 1. --wo 1. " \
                      "-w-occlusion -w-static-occlusion --maxiter 15 -v " \
                      "/media/data/amonszpa/stealth/shared/video_recordings" \
                      "/{scene:s}/skel_{scene:s}_unannot.json independent -s " \
                      "/media/data/amonszpa/stealth/shared" \
                      "/pigraph_scenelets__linterval_squarehist_large_radiusx2_" \
                      "smoothed_withmocap_ct_full_sampling " \
                      "--gap {t0:03d} {t1:03d} --batch-size 15 --dest-dir opt2 " \
                      "--candidates opt1/{g0:03d}_{g1:03d} --n-candidates 200 " \
                      "-tc 0.35".format(scene=name_query, t0=t0, t1=t1, g0=g0, g1=g1)
                print(cmd)
                while g1 < t1+100:
                    p_opt1 = os.path.join(args.video, 'opt1', "%03d_%03d" % (g0, g1))
                    if not os.path.exists(p_opt1):
                        lg.debug("Does not exist: %s" % p_opt1)
                        g0 += 1
                        g1 += 1
                    else:
                        break

                with open(pjoin(args.video, 'gap_command.sh'), 'a') as f2:
                    f2.write(cmd)
                    f2.write("\n")

    # labels
    ax.set_title("Scores and charness w.r.t time: max charness: #%d %g" %
                 (charness_sorted[-1][0], charness_sorted[-1][1]))
    ax.set_xlabel('integer time')
    ax.legend(loc='lower right')
    ax.grid(True)
    ax.yaxis.grid(which='both')
    ax.xaxis.set_ticks(np.arange(T[0]-1, T[-1]+1, 5))
    ax.set_yticks([])
    ax.set_ylim(0., 1.)
    ax.set_ylabel('higher is better')
    plt.suptitle("%s" % name_query)
    with open(os.path.join(d, 'charness_rank.csv'), 'w') as fout:
        fout.write("frame_id,charness\n")
        for fid_charness in reversed(charness_sorted):
            fout.write("{:d},{:g}\n".format(*fid_charness))
            print(fid_charness)
    # plt.show()
    p_out = os.path.join(d, 'charnesses.svg')
    plt.savefig(p_out)
    lg.debug("saved to %s" % p_out)




if __name__ == '__main__':
    show_folder(sys.argv[1:])
