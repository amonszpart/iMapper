import os
import argparse


def argparse_check_exists(path):
    out_path = str(path)
    if not os.path.exists(out_path):
        raise argparse.ArgumentTypeError("%s does not exist" % out_path)
    return out_path
