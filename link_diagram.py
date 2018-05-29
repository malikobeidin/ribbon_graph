from ribbon_graph import *

class EulerianMap(RibbonGraph):
    def __init__(self, opposite, next, PD = []):
        super(EulerianMap,self).__init__(opposite, next, PD = PD)
        for v in self.vertices():
            assert len(v)%2 == 0


class Link(RibbonGraph):
    pass
