__version__ = '1.0'

def version():
    return __version__


from ribbon_graph_base import RibbonGraph, random_link_shadow
from cycle import Path, EmbeddedPath, EmbeddedCycle
from maps import StrandDiagram, Link
from permutation import Bijection, Permutation
from trees import MountainRange, RootedPlaneTree
from decompositions import PolygonWithDiagonals, CycleTree

__all__ = ['RibbonGraph', 'Path', 'EmbeddedPath', 'EmbeddedCycle', 'StrandDiagram', 'Link', 'Permutation', 'Bijection', 'random_link_shadow']
