__version__ = '1.0'

def version():
    return __version__


from ribbon_graph_base import RibbonGraph, random_link_shadow
from cycle import Path, EmbeddedPath, EmbeddedCycle
import maps


__all__ = ['RibbonGraph', 'Path', 'EmbeddedPath', 'EmbeddedCycle']
