import os
import argparse
import sys
from subprocess import call


def main(argv):
    # scene_filepath = '\\\\thorin.cs.ucl.ac.uk/stealth_shared/video_recordings/lobby15'
    # scene_filepath = '\\\\thorin.cs.ucl.ac.uk/stealth_shared/video_recordings/lobby15/opt1b/007_027_charness_0.189568/skel_lobby15_fill_007_027__00.json'
    # scene_filepath = '\\\\thorin.cs.ucl.ac.uk/guerrero/projects/stealth/data/occ_test/0_init.json'
    blender_location = 'C:/Program Files/Blender Foundation/Blender'
    scene_filepath = '\\\\thorin.cs.ucl.ac.uk/guerrero/projects/stealth/data/occ_test/0_opt.json'

    parser = argparse.ArgumentParser(description='Start blender and import scene.')
    parser.add_argument('--scene', type=str, default='scene_filepath', help='Path to the json scene file.')
    parser.add_argument('--out', type=str, default='', help='Path to the output blender file.')
    parser.add_argument('--blender', type=str, default='C:/Program Files/Blender Foundation/Blender',
                        help='Path to the output blender file.')
    parser.add_argument('--fps', type=int, default=10,
                        help='Fps for the scene, this has to match the video fps if a video is given!')
    parser.add_argument('--video', type=str, default='../input/input_cropped_10fps.mp4',
                        help='Path to the input video, relative to the scene file path.')
    parser.add_argument('--quick', action='store_true', help='Quick render with low quality')
    parser.add_argument('-b', action='store_true', help='Background mode')
    parser.add_argument('--candidate', type=str, help='Candidate mode')

    args = parser.parse_args(argv)
    if args.blender:
        blender_location = args.blender

    if len(args.out) == 0:
        args.out = os.path.splitext(args.scene)[0] + '.blend'
        print("will save to %s" % args.out)

    # if os.path.exists(args.out):
    #     os.remove(args.out)

    args.scene = os.path.normpath(args.scene)
    args.out = os.path.normpath(args.out)

    script_filepath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'import_scene.py'))

    if os.path.isdir(blender_location):
        blender_exec = os.path.join(blender_location, 'blender.exe')
    else:
        blender_exec = blender_location
    cmd_params = [
        blender_exec,
        '-noaudio',
        '-P',
        script_filepath,
        '--',
        args.scene,
        '--output',
        args.out,
        '--fps', '{}'.format(args.fps),
        '--video', args.video]

    if args.candidate:
        cmd_params.append('--candidate')
        cmd_params.append(args.candidate)
        
    if args.quick:
        cmd_params.append('--quick')
    if args.b:
        cmd_params.insert(1, '-b')

    call(cmd_params)


if __name__ == '__main__':
    main(sys.argv[1:])
