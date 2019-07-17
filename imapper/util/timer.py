import time


class Timer(object):
    def __init__(self, title=None, verbose=True):
        self.verbose = verbose
        self.title = title

    def __enter__(self):
        self.start = time.time()
        return self

    def get_elapsed_ms(self):
        end = time.time()
        return (end - self.start) * 1000

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('[%s] Elapsed time: %f s' % (self.title, self.secs))