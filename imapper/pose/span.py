class Span(object):
    def __init__(self, start, end, charness, charness_unw,
                 actor_id=0, strict=True, active=True):
        if strict:
            assert start < end
        self.start = start
        self.end = end
        self.charness = charness
        self.charness_unw = charness_unw
        self.smoothed_charness = None
        self.actor_id = actor_id
        self.active = active

    def __repr__(self):
        return 'Span({:d}..{:d}, c={:.3f}{:s})' \
               .format(self.start, self.end, self.charness,
                       ', active' if self.active else '')

    def overlaps(self, other):
        return not (self.end < other.start
                    or self.start > other.end)