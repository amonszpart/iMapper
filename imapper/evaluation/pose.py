import copy
import os
import pickle
import sys
from collections import OrderedDict, namedtuple

import argparse
from enum import IntEnum

import numpy as np
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# https://matplotlib.org/api/font_manager_api.html#matplotlib.font_manager.FontProperties.set_size
plt_params = {
    # 'legend.fontsize': 'x-large',
    #       'figure.figsize': (15, 5),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'
    }
plt.rcParams.update(plt_params)

from imapper.logic.joints import Joint
from imapper.logic.scenelet import Scenelet
from imapper.logic.skeleton import Skeleton
from imapper.pose.main_denis import main as main_denis
from imapper.pose.main_init_path import main as main_init_path
from imapper.util.json import json

p_stealth = os.path.normpath(os.path.abspath(
  os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
))  # type: str

_NAMES = {'name': ('iMapper', 2),
          'tomethreed': ('\\tomethreed', 0),
          'lcrnetthreed$_{no smooth}$': ('\\lcrnetthreed', 1)}
_LOOKUP = {
    'Local': 'LC',
    'World': 'WC',
    '2D': '2D',
    'office1-1-tog-lcrnet': 'office1-1',
    'lobby22-1-tog': 'lobby22-1'
}

# used for time plots
_NAME_LOOKUP ={
    '\\tomethreed': 'Tome3D',
    '\\lcrnetthreed': 'LCR-Net++3D'
}

_SCENE_LOOKUP = {
    'lobby11': 'Scene1',
    'lobby11-couch': 'Scene1',
    'lobby24-3-3': 'Scene2',
    'lobby12': 'Scene3',
    'lobby12-couch-table': 'Scene3',
    'lobby19-3': 'Scene4',
    'lobby18-1': 'Scene5',
    'lobby24-3-1': 'Scene6',
    'lobby15': 'Scene7',
    'lobby24-3-2': 'Scene8',
    'lobby24-2-2': 'Scene9',
    'lobby22-1': 'Scene10',
    'lobby22-1-tog': 'Scene10',
    'library3': 'Scene11',
    'library3-tog': 'Scene11',
    'livingroom00': 'Scene12',
    'office1-1': 'Scene13',
    'office1-1-tog': 'Scene13',
    'office1-1-tog-lcrnet': 'Scene13',
    'garden1': 'Scene14',
    'office2-1': 'Scene15'
}

def _check_exists(l):
    if os.path.exists(l):
        return l
    else:
        l = os.path.join(os.path.dirname(__file__), l)
        if os.path.exists(l):
            return l

    raise RuntimeError("Path does not exist: \"%s\"" % l)

_QUANT_FOLDER = 'quant2'

NamedSolution = namedtuple('NamedSolution', ['name_method', 'path'])


class Rating(object):
    
    @staticmethod
    def rate(frac, metric):
        s = ''
        if metric == 'World':
            if frac < 0.05:
                s = 'EXCELLENT'
            elif frac < 0.1:
                s = 'GOOD'
            elif frac < 0.15:
                s = 'FINE'
            elif frac < 0.2:
                s = 'FAIR'
            else:
                s = 'BAD '
            
        elif metric == 'Local':
            if frac < 10:
                s = 'EXCELLENT'
            elif frac < 15:
                s = 'GOOD'
            elif frac < 20:
                s = 'FINE'
            elif frac < 25:
                s = 'FAIR'
            else:
                s = 'BAD '
        return s
        return '\\texttt{{{}}}'.format(s)
    

class OccludedType(IntEnum):
    OCCLUDED = 0
    VISIBLE = 1
    BOTH = 2

    def __repr__(self):
        return self.__str__().split('.')[-1]
    
    def is_included(self, other):
        if other == OccludedType.BOTH:
            return True
        else:
            return self == other


class Limits(object):
    def __init__(self, min=int(sys.maxsize), max=int(-sys.maxsize)):
        self.min = min
        self.max = max
        
    def merge(self, other):
        if isinstance(other, Limits):
            self.min = min(self.min, other.min)
            self.max = max(self.max, other.max)
        else:
            self.min = min(self.min, other)
            self.max = max(self.max, other)
       
        
Limits2d = namedtuple('Limits2d', ['x', 'y'])


class Series(object):
    def __init__(self, key, frames, values, counts=None, stds=None):
        self.key = key
        self.frames = frames
        self.values = values
        self.counts = counts
        self.stds = stds
        
    def get_limits(self):
        l2d = Limits2d(x=Limits(), y=Limits())
        for x, y in zip(self.frames, self.values):
            l2d.x.merge(x)
            l2d.y.merge(y)
        return l2d


class StatsOverTime(object):
    def __init__(self, plot_fun_name, name_scene):
        self._data = {OccludedType.OCCLUDED: {},
                      OccludedType.VISIBLE: {}}
        self._plot_fun = \
            np.mean if plot_fun_name == 'mean' else \
            np.sum if plot_fun_name == 'sum' else \
            None
        self._name_scene = name_scene
        assert self._plot_fun is not None, \
            'Could not interpret {}'.format(plot_fun_name)
        self._plot_fun_name = plot_fun_name
        
    def add(self, name_method, occluded, title, frame_id, diff):
        l0 = self._data[occluded]
        if title not in l0:
            l0[title] = {}
        l1 = l0[title]
        if name_method not in l1:
            l1[name_method] = {}
        l2 = l1[name_method]
        # assert frame_id not in l1[name_method], l1[name_method]
        if frame_id not in l2:
            l2[frame_id] = []
        l3 = l2[frame_id]
        l3.append(diff)
        
    def paint(self, path_dest):
        series = OrderedDict()
        for occluded, l0 in sorted(self._data.items()):
            for metric, l1 in sorted(l0.items()):
                for name_method in sorted(l1, key=lambda k: _NAMES[k[1:]][1]):
                    l2 = l1[name_method]
                    keys = []
                    # Mean per frame
                    values = []
                    counts = []
                    stds = []
                    # Sum per frame, without mean
                    for key in sorted(l2.keys()):
                        keys.append(key)
                        values.append(self._plot_fun(l2[key]))
                        counts.append(len(l2[key]))
                        
                    # insert
                    if occluded not in series:
                        series[occluded] = OrderedDict()
                    if metric not in series[occluded]:
                        series[occluded][metric] = []
                    series[occluded][metric].append(
                        Series(key=name_method,
                               frames=keys, values=values, counts=counts,
                               stds=None)
                    )
                    
        # get maxima
        maxima_x = Limits()
        maxima_y = []
        for occluded, l1 in series.items():
            for col, (metric, l2) in enumerate(l1.items()):
                if len(maxima_y) <= col:
                    maxima_y.append(Limits())
                for serie in l2:
                    l2d = serie.get_limits()
                    maxima_x.min = min(l2d.x.min, maxima_x.min)
                    maxima_x.max = max(l2d.x.max, maxima_x.max)
                    maxima_y[col].min = min(maxima_y[col].min, l2d.y.min)
                    maxima_y[col].max = max(maxima_y[col].max, l2d.y.max)
                    
        # Plot
        fig = plt.figure(figsize=(18, 6))
        for entry in (OccludedType.OCCLUDED, OccludedType.VISIBLE):
            if entry not in series:
                continue
            for col, metric in enumerate(sorted(series[entry])):
                serie_lst = series[entry][metric]
                ax = fig.add_subplot(230 + entry * 3 + col + 1)
                info_text = 'Per-frame mean:'
                for serie in serie_lst:
                    label = '{}'.format(_NAMES[serie.key[1:]][0])
                    ax.plot(serie.frames, serie.values,
                            label=label,
                            linestyle='-', marker='.')
                    info_text_ = '{}: {:.2f}'.format(label, np.mean(serie.values))
                    info_text = '{}\n  {}'.format(info_text, info_text_) \
                        if len(info_text) else info_text_
                    
                ax.set_title(metric)
                if col == 0:
                    ax.set_ylabel(entry.__repr__())
                if entry == 0 and col == 2:
                    ax.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.3))
                ax.set_xlim(maxima_x.min, maxima_x.max)
                ax.set_ylim(maxima_y[col].min, maxima_y[col].max)
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                # place a text box in upper left in axes coords
                ax.text(0.05, 0.95, info_text,
                        transform=ax.transAxes,
                        verticalalignment='top', bbox=props)
        fig.suptitle('{} -- {}'.format(
            self._name_scene, self._plot_fun_name.capitalize()))
        plt.subplots_adjust(left=0.04, bottom=0.05, right=0.98)
        plt.savefig(os.path.join(path_dest,
                                 'stats_{}.pdf'.format(self._plot_fun_name)))
        
        return series


class Comparison(object):
    _axis_titles = ['x', 'y', 'z']
    _view_titles = {(0, 1): 'front', (0, 2): 'top'}
    
    def __init__(self, title, path_scene, name_method, stats):
        self._title = title
        self._data = []
        self._path_scene = path_scene
        self._name_method = name_method
        self._p_out = os.path.join(self._path_scene,
                                   'debug_eval')
        if not os.path.isdir(self._p_out):
            os.makedirs(self._p_out)
            
        self._stats = stats
        self.dimensions = np.zeros((3,))

    @property
    def title(self):
        return self._title

    def add(self, entry, gt, occluded, frame_id, scale=1.):
        assert entry.shape[0] <= 3 and gt.shape[0] <= 3, (entry, gt)
        diff = np.linalg.norm(entry - gt, axis=0) * scale
        
        sub = []
        for j in range(Joint.get_num_joints()):
            if j in (Joint.PELV, Joint.NECK):
                continue
            if occluded[j]:
                sub.append(diff[j])
                self._stats.add(occluded=OccludedType.OCCLUDED,
                                title=self._title,
                                name_method=self._name_method,
                                frame_id=frame_id,
                                diff=diff[j])
            else:
                self._stats.add(occluded=OccludedType.VISIBLE,
                                title=self._title,
                                name_method=self._name_method,
                                frame_id=frame_id,
                                diff=diff[j])
        self._data.extend(sub)

    def rmsq(self):
        data = np.array(self._data)
        return np.sqrt(np.mean(np.square(data)))
        # return np.mean(np.abs(data))


def evaluate(named_solution, sclt_gt, sclt_gt_2d, frame_ids,
             path_scene, stats, actions=None, scale=100.):
    """
    
    :param named_solution:
    :param sclt_gt:
    :param sclt_gt_2d:
    :param frame_ids:
    :param path_scene:
    :param stats:
    :param actions:
    :param scale: scale from meter to cm
    :return:
    """
    p_intrinsics = os.path.join(path_scene, 'intrinsics.json')
    intrinsics = np.array(json.load(open(p_intrinsics, 'r')),
                          dtype=np.float32)
    print('Loading {}'.format(named_solution.path))
    sclt_sol = Scenelet.load(named_solution.path)
    sclt_sol.skeleton._visibility.clear()
    sclt_sol.skeleton._confidence.clear()
    sclt_sol.skeleton._forwards.clear()
    sclt_sol.skeleton = Skeleton.resample(sclt_sol.skeleton)
    err_3d = Comparison(title='World',
                        path_scene=path_scene,
                        name_method=named_solution.name_method,
                        stats=stats)
    err_3d_local = Comparison(title='Local',
                              path_scene=path_scene,
                              name_method=named_solution.name_method,
                              stats=stats)
    err_2d = Comparison(title='2D',
                        path_scene=path_scene,
                        name_method=named_solution.name_method,
                        stats=stats)
    
    occlusion = sclt_gt.aux_info['occluded']
    missing = {'method': [], 'gt': []}
    for frame_id in frame_ids:
        try:
            entry = sclt_sol.skeleton.get_pose(frame_id=frame_id)
        except KeyError:
            missing['method'].append(frame_id)
            continue
        if actions is not None and frame_id in actions \
          and actions[frame_id] == 'walking':
            print('Skipping non-interactive frame {} {}'
                  .format(frame_id, actions[frame_id]))
            continue

        # 3D
        gt = sclt_gt.skeleton.get_pose(frame_id=frame_id)
        occluded = occlusion['{:d}'.format(frame_id)]
        err_3d.add(entry=entry, gt=gt, frame_id=frame_id, scale=scale,
                   occluded=occluded)

        # Local 3D
        local_entry = entry - entry[:, Joint.PELV:Joint.PELV+1]
        local_gt = gt - gt[:, Joint.PELV:Joint.PELV+1]
        err_3d_local.add(entry=local_entry, gt=local_gt, frame_id=frame_id,
                         scale=scale, occluded=occluded)

        #
        # GT 2D
        #

        gt_2d = sclt_gt_2d.skeleton.get_pose(frame_id=frame_id)
        entry_2d = entry[:2, :] / entry[2, :]
        entry_2d[0, :] *= intrinsics[0, 0]
        entry_2d[1, :] *= intrinsics[1, 1]
        entry_2d[0, :] += intrinsics[0, 2]
        entry_2d[1, :] += intrinsics[1, 2]

        err_2d.add(entry=entry_2d, gt=gt_2d[:2, :], frame_id=frame_id,
                   occluded=occluded)

    # stats.paint(path_dest=os.path.join(path_scene, 'debug_eval'))
    mn, mx = np.min(sclt_gt.skeleton.poses, axis=(0, 2)), \
             np.max(sclt_gt.skeleton.poses, axis=(0, 2))
    err_3d.dimensions = (mx - mn) * scale

    assert len(missing['method']) < len(frame_ids)/2, (missing, frame_ids)
    
    return OrderedDict({
        err_3d.title: err_3d,
        err_3d_local.title: err_3d_local,
        err_2d.title: err_2d,
        '_missing': missing
    })


def two_row(a, b):
    return '\\begin{{tabular}}{{@{{}}c@{{}}}}' \
           ' {} \\\\ ' \
           ' {} ' \
           '\\end{{tabular}}'.format(a, b)


def get_interaction_spans(actions):
    srtd = sorted(((k, v) for k, v in actions.items()),
                  key=lambda kv: kv[0])
    spans = []
    span = [None, None]
    for i, (frame, action) in enumerate(srtd):
        if action == 'walking':
            if span[0] is not None:
                spans.append(tuple((span[0], frame)))
                span = [None, None]
        else:
            if span[0] is None:
                span[0] = frame
    if span[0] is not None:
        spans.append(tuple((span[0], frame)))
        
    return spans


def find_best(visibilities, spans):
    best = {}
    for visibility, methods in visibilities.items():
        if visibility not in best:
            best[visibility] = {}
        for method, metrics in methods.items():
            for metric, serie in metrics:
                if metric not in best[visibility]:
                    best[visibility][metric] = {}
                for frame, value in zip(serie.frames, serie.values):
                    if spans is not None:
                        span = next(((start, end) for start, end in spans
                                     if start <= frame <= end), None)
                        if span is None:
                            continue
                        else:
                            print('Not skipping {} becausea of {}'.format(frame, span))
                    val_method = best[visibility][metric].get(frame)
                    if val_method is None or value < val_method[0]:
                        best[visibility][metric][frame] = (value, method)
    counts = {}
    for visibility, metrics in best.items():
        counts[visibility] = {}
        for metric, frames in metrics.items():
            assert metric not in counts[visibility], "?"
            counts[visibility][metric] = {}
            for frame, (value, method) in frames.items():
                if method not in counts[visibility][metric]:
                    counts[visibility][metric][method] = 0
                # if value < 0.1:
                counts[visibility][metric][method] += 1
            for method in counts[visibility][metric]:
                counts[visibility][metric][method] /= float(len(frames))
            
    print(counts)
    return counts
    
     
def plot_over_time(lines, all_actions):
    VIS_END = 1
    COLS = 2
    ROWSPAN = 3
    for name_scene, visibilities in lines:
        x_min = min((min(serie.frames)
                     for methods in visibilities.values()
                     for metrics in methods.values()
                     for metric, serie in metrics))
        x_max = max((max(serie.frames)
                     for methods in visibilities.values()
                     for metrics in methods.values()
                     for metric, serie in metrics))
        actions = all_actions.get(name_scene)
        spans = None
        if actions is not None:
            spans = get_interaction_spans(actions)
        counts = find_best(visibilities, spans)
        
        for the_metric in ('World', 'Local'):
            plt.figure(figsize=(12, 4), dpi=150)

            y_max = max((max(serie.values)
                         for methods in visibilities.values()
                         for metrics in methods.values()
                         for metric, serie in metrics
                         if metric == the_metric))
            if the_metric == 'World':
                y_max = min(y_max, .35)
            
            for vis in range(VIS_END):
                # ax = plt.subplot2grid((2, 1), (vis, 0))
                ax1 = plt.subplot2grid((VIS_END * ROWSPAN + 1, COLS), (vis, 0), colspan=2, rowspan=ROWSPAN)
                ax2 = None
                legend_elems = []
                counts_srtd = sorted(((method, count) for method, count in counts[vis][the_metric].items()),
                                     key=lambda e: e[1], reverse=True)
                info_text = '\n'.join(
                    ' {}: {:.1f}%'.format(
                        _NAME_LOOKUP[method]
                        if method in _NAME_LOOKUP
                        else method,
                        count * 100.)
                    for method, count in counts_srtd)
                info_text = 'Best per-frame:\n{}'.format(info_text)
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                # ax1.text(0.025, 0.975, info_text,
                #          transform=ax1.transAxes,
                #          verticalalignment='top', bbox=props)
                for visibility, methods in visibilities.items():
                    if not OccludedType(visibility).is_included(vis):
                        continue

                    ax = None
                    prop_cycler = None
                    for method_id, (method, metrics) in enumerate(methods.items()):
                        for metric, serie in metrics:
                            if metric != the_metric:
                                continue
                                
                            if method_id == 0:
                                ax1.set_xlim(x_min, x_max)
                                ax1.set_ylim(0, y_max)
                            if ax2 is None:
                                # ax2 = ax1.twinx()
                                ax2 = plt.subplot2grid((VIS_END * ROWSPAN + 1, COLS), (VIS_END * ROWSPAN, 0), colspan=2)
                                legend_elems.extend(
                                    ax2.plot(serie.frames, serie.counts,
                                             '.--', c='k',
                                             label='#joints occluded'))
                                ax2.set_ylim(0, 11)
                                ax2.set_ylabel('#joints\noccluded')
                                ax2.set_xlabel('Frame ID')
                                ax2.set_xlim(x_min, x_max)
                                prev_label = None
                                for fr, cn in zip(serie.frames, serie.counts):
                                    if prev_label is None \
                                      or prev_label != cn:
                                        ax2.text(fr, cn+1., '{}'.format(cn),
                                                 fontsize='medium')
                                        prev_label = cn
                                
                            if COLS > 2:
                                ax = plt.subplot2grid(
                                    (VIS_END, COLS), (vis, 2 + method_id),
                                    sharey=ax1)
                            if prop_cycler is None:
                                prop_cycler = ax1._get_lines.prop_cycler
                            color = next(prop_cycler)['color']
                            name_method = _NAME_LOOKUP.get(method)
                            if name_method is None:
                                name_method = method
                            legend_elems.extend(
                                ax1.plot(serie.frames, serie.values,
                                         '.--', c=color, label=name_method))
                            ax1.grid(b=True, which='major', axis='y')

                            if ax is not None:
                                ax.plot(serie.frames, serie.values,
                                        '.--', c=color, label=name_method)
                                ax.grid(b=True, which='major', axis='y')
                                
                            if method_id == 0:
                                ylabel = 'Avg.err.(cm)' \
                                    if the_metric == 'Local' \
                                    else 'Avg.err./scn.diam.'
                                if ax is not None:
                                    ax.set_ylabel(ylabel)
                                if ax1 is not None:
                                    ax1.set_ylabel(ylabel)
                            if vis == VIS_END - 1:
                                if ax is not None:
                                    ax.set_xlabel('Frame ID')
                            elif vis == 0:
                                if ax is not None:
                                    ax.set_title(name_method)
                                
                            if spans is not None:
                                for st, en in spans:
                                    if ax is not None:
                                        ax.axvspan(st, en, facecolor='g', alpha=0.25)
                                    if method_id == 0:
                                        ax1.axvspan(st, en, facecolor='g', alpha=0.25)
                            if ax is not None:
                                ax.set_xlim(x_min, x_max)
                                ax.set_ylim(0, y_max)
                                # ax.set_ylim(0, y_max[vis])
                legend_elems = legend_elems[1:] + legend_elems[:1]
                ax1.legend(handles=legend_elems, loc='upper left')
                ax1.set_xticklabels([])
                
            p_out = '/Users/aron/workspace/ucl/stealth/data/video_recordings/rebuttal/'
            if not os.path.isdir(p_out):
                os.makedirs(p_out)
            p_out = os.path.join(p_out, '{}_{}.png'.format(name_scene, the_metric))
            print(p_out)
            suptitle = name_scene
            if name_scene in _SCENE_LOOKUP:
                suptitle = _SCENE_LOOKUP[name_scene]
            else:
                print('Could not find {}'.format(name_scene))
            plt.suptitle("{}".format(suptitle), fontsize='xx-large')
            plt.subplots_adjust(left=0.08, right=0.95, bottom=0.16, top=0.9,
                                wspace=0.12, hspace=0.8)
            plt.gcf().text(
                0.0835, 0.94, the_metric + '; ' + OccludedType(vis).__repr__() + ' joints', fontsize=16,
                bbox=dict(facecolor='black', alpha=0.05, capstyle='round',
                          edgecolor='white'))
            plt.gcf().text(
                0.845, 0.94, 'Interacting', fontsize=16,
                bbox=dict(facecolor='g', alpha=0.2, capstyle='round',
                          edgecolor='white'))
            plt.savefig(p_out)
            p_out = os.path.join(os.path.dirname(p_out), '{}_{}.pdf'
                                 .format(name_scene, the_metric))
            plt.savefig(p_out)
            plt.close()
    
    print('plot_over_time over')
    # sys.exit(1)


def table2(summary, type_vis, bounds, all_actions=None):
    docamble = '\\documentclass[8pt,a4paper]{article}\n' \
               '\\usepackage[latin1]{inputenc}\n' \
               '\\usepackage{amsmath}\n' \
               '\\usepackage{amsfonts}\n' \
               '\\usepackage{amssymb}\n' \
               '\\usepackage{graphicx}\n' \
               '\\usepackage[landscape]{geometry}\n' \
               '\\newcommand{\\lcrnetthreed}{\\mbox{LCR-Net++3D}}\n' \
               '\\newcommand{\\tomethreed}{\\mbox{Tome3D}}' \
               '\\newcommand{\\Scene}[1]{#1}' \
               '\\begin{document}\n'
    
    preamble = '\\setlength{\\tabcolsep}{0.11em}\n' \
               '\\begin{table*}[h!]\n' \
               ' \\scriptsize\n' \
               ' \\caption{Comparison of skeleton estimates:' \
               ' rating, fraction of 2D scene diagonal (WC) or error in cm (LC)' \
               ' in brackets: (how much worse, mean, std).}\n' \
               ' \\centering\n' \
               ' \\begin{tabular}{|l|'
    lines = []
    lines_per_frame = []
    best = {}
    cnt = None
    s = ''
    typs = None
    metrics = None
    viss = None  # visibilities
    for (name_scene, occlusion_threshold), series in summary.items():
        bounds_ = bounds[name_scene]
        bounds_scale = np.sqrt(bounds_[0]**2 + bounds_[2]**2)
        best[name_scene] = {}
        line = OrderedDict()
        line_per_frame = OrderedDict()
        cnt_ = 0
        metrics_ = []
        vis_ = []
        for visibility in (OccludedType.OCCLUDED, OccludedType.VISIBLE):
            visib_to_table = visibility.is_included(type_vis)
            if visib_to_table:
                line[visibility] = OrderedDict()
                vis_.append(visibility.__repr__())
            line_per_frame[visibility] = OrderedDict()
            for col, metric in enumerate(sorted(series[visibility])):
                # ignoring 2D
                if metric == '2D':
                    continue
                if visib_to_table:
                    typs_ = []
                    metrics_.append(metric)
                serie_lst = series[visibility][metric]
                for serie in serie_lst:
                    label = '{}'.format(_NAMES[serie.key[1:]][0])
                    avg = np.mean(serie.values)
                    mn = np.min(serie.values)
                    mx = np.max(serie.values)
                    std = np.std(serie.values)
                    
                    # error as fraction of scene bounds
                    frac = avg / bounds_scale
                    
                    info_text_ = '{} {}: {:.1f} \tmn: {:.1f} mx:{:.1f}, ' \
                                 'std: {:.1f}, frac: {:.3f}' \
                        .format(name_scene, label, avg, mn, mx, std, frac)
                    print(name_scene, visibility, metric, info_text_)
                    
                    value_ = frac if metric == 'World' else avg
                    # for paper table (one table per visibility)
                    if visib_to_table:
                        cnt_ += 1
                        typs_.append(label)
                        key_best = (visibility, metric)
                        if key_best not in best[name_scene] \
                          or value_ < best[name_scene][key_best][1]:
                            best[name_scene][key_best] = (label, value_)
                            
                        try:
                            line[visibility][label].append(
                                tuple((metric, (value_, avg, std, len(serie.values)))))
                        except KeyError:
                            line[visibility][label] = [
                                (metric, (value_, avg, std, len(serie.values)))]
                            
                    # for final version plotting (both visibilities)
                    serie_cpy = copy.deepcopy(serie)
                    if metric == 'World':
                        serie_cpy.values /= bounds_scale
                    try:
                        line_per_frame[visibility][label].append(
                            tuple((metric, serie_cpy))
                        )
                    except KeyError:
                        line_per_frame[visibility][label] = [
                            tuple((metric, serie_cpy))]
                        
                if visib_to_table and typs is None:
                    typs = typs_
        
        # if visib_to_table:
        if cnt is None:
            cnt = cnt_
        if metrics is None:
            metrics = metrics_
        if viss is None:
            viss = vis_
        lines.append((name_scene, line))
            
        lines_per_frame.append((name_scene, line_per_frame))
    if type_vis == OccludedType.OCCLUDED:
        plot_over_time(lines_per_frame, all_actions=all_actions)

    method_avgs = {}
    for name_scene, entries in lines:
        s += two_row('\\Scene{{{}}} ' \
                     .format(name_scene
                             if name_scene not in _LOOKUP
                             else _LOOKUP[name_scene]),
                     '\\ ')
        assert len(entries) == 1, "Assumed single visibility"
        for visibility, line in entries.items():
            for label, avgs in line.items():
                if label not in method_avgs:
                    method_avgs[label] = {}
                for metric, avg_entries in avgs:
                    if metric not in method_avgs[label]:
                        method_avgs[label][metric] = [0., 0]
    
                    avg = avg_entries[0]
                    n_entries = avg_entries[3]
                    
                    method_avgs[label][metric][0] += avg * n_entries
                    method_avgs[label][metric][1] += n_entries
                    
                    aux = ', '.join('{:.1f}'.format(e)
                                    if e > 1 else '{:.3f}'.format(e)[1:]
                                    for e in avg_entries[1:])
                    rating = Rating.rate(avg, metric)
                    # snum = '\\texttt{{{}}}'.format(
                    snum = '{:.3f}'.format(avg)[1:] \
                        if avg <= 1 else '{:.1f}'.format(avg)
                    if best[name_scene][(visibility, metric)][0] == label:
                        sperf = ''
                        # s = '{}\t{}'.format(
                        #     s, '& \\textbf{{{} {}}} ({}\\if0, {}\\fi)'.format(
                        #         rating, snum, sperf, aux))
                        s = '{}\t{}'.format(
                            s,
                            '& {}'.format(
                                two_row(
                                    '\\textbf{{{}}} {}'.format(rating, snum)
                                    if rating == 'GOOD'
                                       or rating == 'EXCELLENT' else
                                    '{} {}'.format(rating, snum),
                                    '\\ ')
                            )
                        )
                    else:
                        best_val = best[name_scene][(visibility, metric)][1]
                        perf = (avg - best_val) / best_val
                        sperf = '+{:.0f}\%'.format(perf * 100.) \
                            if perf > 0.005 \
                            else '+{:.1f}\%'.format(perf * 100.)
                        # s = '{}\t{}'.format(
                        #     s, '& {} {} ({}\\if0, {}\\fi)'.format(
                        #         rating, snum, sperf, aux))
                        s = '{}\t{}'.format(
                            s,
                            '& {}'.format(
                                two_row(
                                    '{} {}'.format(
                                        '\\textbf{{{}}}'.format(rating)
                                        if rating == 'GOOD' or rating == 'EXCELLENT'
                                        else rating,
                                        snum),
                                    '({{\\scriptsize {}}})'.format(sperf)))
                        )
        s += ' \\\\\n'
     
    s += '\\hline\\hline\\\\\n$\\mu$ '
    for label, metrics in method_avgs.items():
        for metric, (avg_sum, avg_cnt) in metrics.items():
            s = '{:s}\t& {:.3f}'.format(s, avg_sum / avg_cnt)
    s += ' \\\\\n'
        
    preamble = '{}{}}}'.format(preamble, cnt * 'c|')

    # visibility (visible, occluded)
    visibility_line = '  '
    for visibility in viss:
        visibility_line = '{prev:s}&\t' \
                      '\\multicolumn{{6}}{{|c|}}{{{{{visibility:s}}}}}\t' \
            .format(prev=visibility_line, visibility=visibility)
        
    # name of method (Tome, LCRNet, iMapper)
    typ_line = '  '
    for visbility in viss:
        for typ in typs:
            typ_line = '{prev:s}&\t' \
                       '\\multicolumn{{2}}{{|c|}}{{{{{typ:s}}}}}\t' \
                .format(prev=typ_line, typ=typ)
        
    # metrics (2D, LC, WC)
    metric_line = '  '
    for typ in typs:
        for metric in metrics:
            metric_line = '{prev:s}&\t' \
                          '\\multicolumn{{1}}{{|c|}}{{{{{metric:s}}}}}\t' \
                .format(prev=metric_line, metric=_LOOKUP[metric])
        
    s = '{}\n' \
        '  \\hline\n' \
        '{}\\\\\n\\hline\n' \
        '{}\\\\\n\\hline\n' \
        '{}\\\\\n\\hline\n' \
        '{}'\
        .format(docamble + preamble, visibility_line, typ_line, metric_line, s)
    s = '{}' \
        '\\hline\n' \
        ' \\end{{tabular}}\n' \
        ' \\label{{t:poses}}\n' \
        '\\end{{table*}}\n'.format(s)
    s += '\n\\end{document}'
    with open('/tmp/auto.tex', 'w') as f:
        f.write(s)
    print(s)
    cmd = 'pdflatex /tmp/auto.tex; open -a Preview auto.pdf'
    os.system(cmd)


def table(summary):
    preamble = '\setlength{\\tabcolsep}{0.11em}\n' \
               '\\begin{table*}[h!]\n' \
               ' \\small\n' \
               ' \\caption{{Comparison of skeleton estimates.}}\n' \
               ' \\centering\n' \
               ' \\begin{tabular}{|l|'
    methods = None
    typs = None
    s = ''
    for (name_scene, occlusion_threshold), errors in summary.items():
        # scene title
        s = '{s:s}\t{{\\Scene{{{scene:s}}}$_{{{ot:d}}}$}}\t' \
            .format(s=s, scene=name_scene, ot=occlusion_threshold)

        best = {'rmsq': {}}
        for name_method, comp in errors.items():
            for typ, error in comp.items():
                if typ.startswith('_'):
                    continue
                err_rmsq = error.rmsq()
                if typ in best['rmsq']:
                    best['rmsq'][typ] = min(best['rmsq'][typ], err_rmsq)
                else:
                    best['rmsq'][typ] = err_rmsq

        # scores
        methods_ = []
        for name_method, comp in errors.items():
            methods_.append(name_method)
            typs_ = []
            for typ, error in comp.items():
                if typ.startswith('_'):
                    continue
                typs_.append(typ)
                print('{}{}: {}'.format(name_method, typ, err_rmsq))

                err_rmsq = error.rmsq()
                if err_rmsq == best['rmsq'][typ]:
                    s_entry = '\\textbf{{{err:.2f}}}'.format(err=err_rmsq)
                else:
                    s_entry = '{err:.2f}'.format(err=err_rmsq)

                s = '{s:s}&\t{{{entry:s}}}\t' \
                    .format(s=s, entry=s_entry)
            if typs is None:
                typs = typs_
            else:
                assert all(a == b for a, b in zip(typs, typs_)), \
                    (typs, typs_)

        # linebreak
        s = '{}\\\\\n'.format(s)

        if methods is None:
            methods = methods_
        else:
            assert all(a == b for a, b in zip(methods, methods_)), \
                (methods, methods_)

    typ_line = '  '
    for typ in typs:
        typ_line = '{prev:s}&\t' \
                   '\\multicolumn{{1}}{{|c|}}{{{{{typ:s}}}}}\t' \
            .format(prev=typ_line, typ=typ)

    typs_line = ''
    preamble = '{}{}}}\n' \
               '  \\hline \n' \
               '  '.format(preamble,
                           len(methods) * ('{}|'.format(len(typs) * 'c')))
    for method in methods:
        preamble = '{}{}'.format(
          preamble,
          '&\t\\multicolumn{{{n_typ:d}}}{{|c|}}{{{{{method:s}}}}}\t'
              .format(n_typ=len(typs), method=method))

        typs_line = '{}{}'.format(typs_line, typ_line)

    preamble = '{}\\\\\n  \\hline\n{}\\\\\n'.format(preamble, typs_line)
    s = '{}' \
        '  \\hline\n' \
        '{}'.format(preamble, s)
    
    # Add mean line at end
    # for name_method, avg_typs in method_avgs:
    #     for avg_typ, (avg_sum, avg_cnt) in avg_typs:
    #       s = '{s:s}&\t{{{entry:.2f}}}\t' \
    #           .format(s=s, entry=avg_sum / avg_cnt)
            
    s = '{}' \
        ' \\hline\n' \
        ' \\end{{tabular}}\n' \
        ' \\label{{t:poses}}\n' \
        '\\end{{table*}}\n'.format(s)
    print(s)
    with open('/home/amonszpa/workspace/paper-stealth/interactionAssisted'
              '/062_quant_joints_auto.tex', 'w') as f:
        f.write(s)


def read_actions(p_root):
    p_actions = os.path.join(p_root, 'gt', 'actions.txt')
    actions = None
    if os.path.isfile(p_actions):
        with open(p_actions, 'r') as f:
            actions = [line.strip().split('\t') for line in f.readlines()]
            actions = {int(line[0]): line[1] for line in actions}
    return actions


def work_scene(p_root, plot_fun_name, name_scene, fname):
    pjoin = os.path.join

    name_scene = os.path.split(p_root)[-1]
    p_actions = os.path.join(p_root, 'gt', 'actions.txt')
    actions = None
    # if os.path.isfile(p_actions):
    #     with open(p_actions, 'r') as f:
    #         actions = [line.strip().split('\t') for line in f.readlines()]
    #         actions = {int(line[0]): line[1] for line in actions}

    p_quant2 = pjoin(p_root, _QUANT_FOLDER)
    if not os.path.isdir(p_quant2):
        os.makedirs(p_quant2)

    p_methods = []

    #
    # Tome3D
    #
    
    pfixes = [('Tome3D-nosmooth', '0')]  # ('Tome3D-smooth', '10')
    for postfix, smooth in pfixes:
        p_tome3d = pjoin(p_quant2,
                         'skel_{}_{}.json'.format(name_scene, postfix))
        p_tome_skels = pjoin(p_root, 'denis')
        if os.path.isfile(p_tome3d):
            p_methods.append(NamedSolution('\\tomethreed', p_tome3d))
        else:
            print('Can\'t find Tome3D at {}'.format(p_tome3d))
        
    #
    # LCR-Net 3D
    #

    postfix = 'LCRNet3D-nosmooth'
    p_lcrnet3d = pjoin(p_quant2,
                       'skel_{}_{}.json'.format(name_scene, postfix))
    if os.path.isfile(p_lcrnet3d):
        p_methods.append(
            NamedSolution('\\lcrnetthreed$_{no smooth}$', p_lcrnet3d))
    else:
        print('Can\'t find Tome3D at {}'.format(p_tome3d))
    
    #
    # GT
    #

    p_gt = pjoin(p_root, 'gt', 'skel_{}_GT.json'.format(name_scene))
    assert os.path.isfile(p_gt), 'Need gt file: {}'.format(p_gt)
    sclt_gt = Scenelet.load(p_gt)

    p_gt_2d = pjoin(p_root, 'gt', 'skel_GT_2d.json')
    sclt_gt_2d = Scenelet.load(p_gt_2d)

    #
    # Evaluate
    #

    p_methods.append(
      NamedSolution('\\name',
                    pjoin(p_root, 'output', fname)))
    
    # Append your solution here
    
    p_methods.append(
        NamedSolution('NewMethod',
                      pjoin(p_root, 'NewMethod', fname)))

    stats = StatsOverTime(plot_fun_name=plot_fun_name, name_scene=name_scene)
    errors = OrderedDict()
    for named_solution in p_methods:
        frame_ids = [fid for fid in sclt_gt.skeleton.get_frames()
                     if not fid % 2]
        # print(frame_ids)
        errors_ = evaluate(named_solution=named_solution, sclt_gt=sclt_gt,
                           sclt_gt_2d=sclt_gt_2d, frame_ids=frame_ids,
                           path_scene=p_root, stats=stats, actions=actions)
        errors[named_solution.name_method] = errors_
    series = stats.paint(path_dest=os.path.join(p_root, 'debug_eval'))
    return errors, series


def main(argv):
    pjoin = os.path.join

    parser = argparse.ArgumentParser("")
    parser.add_argument('d', type=_check_exists,
                        help="Input directory")
    parser.add_argument('--func', choices=('sum', 'mean'), default='mean')
    parser.add_argument('--fname', type=str, default='skel_output.json')
    args = parser.parse_args(argv)
    # plot_fun = \
    #     np.mean if args.func == 'mean' else \
    #     np.sum if args.func == 'sum' else \
    #     None
    print("Working with %s" % args.d)
   
    all_actions = {}
    p_pickle = os.path.join(args.d, 'summary2.pickle')
    if True or not os.path.isfile(p_pickle):
        summary = OrderedDict()
        summary2 = OrderedDict()
        scenes = ['lobby19-3', 'lobby18-1', 'lobby15', 'lobby22-1-tog',
                  'livingroom00', 'office1-1-tog-lcrnet', 'library3-tog',
                  'garden1']
        bounds = {}
        for name_scene in scenes:
            p_root = pjoin(args.d, name_scene)
            errors, series = work_scene(
                p_root=p_root,
                plot_fun_name=args.func,
                name_scene=name_scene,
                fname=args.fname)
            key = name_scene
            assert key not in summary
            summary[key] = errors
            assert name_scene not in bounds
            bounds[name_scene] = errors['\\name']['World'].dimensions
            summary2[key] = series
            actions = read_actions(p_root)
            if actions is not None:
                all_actions[name_scene] = actions
       
        pickle.dump((summary2, bounds, all_actions), open(p_pickle, 'wb'))
        print('Wrote to {}'.format(p_pickle))
    else:
        print('Reading from {}'.format(p_pickle))
        summary2, bounds, all_actions = pickle.load(open(p_pickle, 'rb'))

    table2(summary2, OccludedType.OCCLUDED, bounds, all_actions=all_actions)
    # table2(summary2, OccludedType.VISIBLE, bounds)


if __name__ == '__main__':
    main(sys.argv[1:])
