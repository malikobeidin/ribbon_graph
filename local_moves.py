

def add_edge(ribbon_graph, label1, label2):
    """
    Add a new edge
    """
    pass

def cut_edge(ribbon_graph, label):
    """
    Disconnect an edge, but keep the half-edge labels on each side.
    """
    ribbon_graph.opposite.undo_two_cycle(label)


def delete_edge(ribbon_graph, label):
    """
    Delete the entire edge, with the half-edge labels.
    """
    op_label = ribbon_graph.opposite[label]
    ribbon_graph.opposite.remove_cycle(label)
    
    ribbon_graph.next.split_label_from_cycle(label)
    ribbon_graph.next.split_label_from_cycle(op_label)
    ribbon_graph.next.remove_fixed_point(label)
    ribbon_graph.next.remove_fixed_point(op_label)


def contract_edge(ribbon_graph, label):
    """
    Merge the vertices on either side of the edge by collapsing the edge.
    """
    op_label = ribbon_graph.opposite[label]

    label_previous = ribbon_graph.next.previous(label)
    op_label_previous = ribbon_graph.next.previous(op_label)

    delete_edge(ribbon_graph, label)
    ribbon_graph.next.merge_cycles_at(label_previous, op_label_previous)





