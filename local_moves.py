
    

def add_edge(ribbon_graph, label1, label2, new_label1, new_label2):
    """
    Add a new edge connecting the corners AFTER label1 and label2, with
    the new_edge [new_label1, new_label2]
    """
    ribbon_graph.next.insert_after(label1, new_label1)
    ribbon_graph.next.insert_after(label2, new_label2)
    ribbon_graph.opposite.add_cycle([new_label1, new_label2])

def double_edge(ribbon_graph, label, new_label1, new_label2):
    op_label = ribbon_graph.opposite[label]
    prev = ribbon_graph.vertex(op_label)[-1]
    add_edge(ribbon_graph, label, prev, new_label1, new_label2)

def split_vertex(ribbon_graph, label1, label2):    
    ribbon_graph.next.split_cycle_at(label1, label2)

def merge_vertices(ribbon_graph, label1, label2):    
    ribbon_graph.next.merge_cycles_at(label1, label2)

    
def connect_edges(ribbon_graph, label1, label2):
    """
    
    """
    if ribbon_graph.opposite[label1] == label1 and ribbon_graph.opposite[label2] == label2:
        ribbon_graph.opposite.merge_cycles_at(label1, label2)
    else:
        raise Exception("Edges already connected.")

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

def cut_vertex(ribbon_graph, label):
    """
    Cut all edges coming out of a vertex.
    """
    seen_labels = set([])
    vertex = ribbon_graph.vertex(label)
    for l in vertex:
        if l not in seen_labels:
            seen_labels.add(l)
            seen_labels.add(ribbon_graph.opposite[l])
            ribbon_graph.opposite.undo_two_cycle(l)
    for l in vertex:
        ribbon_graph.opposite.remove_fixed_point(l)
    ribbon_graph.next.remove_cycle(label)


def delete_vertex(ribbon_graph, label):
    seen_labels = set([])
    vertex = ribbon_graph.vertex(label)
    for l in vertex:
        if l not in seen_labels:
            seen_labels.add(l)
            seen_labels.add(ribbon_graph.opposite[l])
            delete_edge(ribbon_graph, l)

    
    
def contract_face(ribbon_graph, label):
    """
    If the face if embedded (that is, no vertex is encountered twice when
    going around the face), then it is topologically a disk. This function
    collapses the disk to a single vertex.

    If the face is not embedded, then this will result in an error.
    """
    
    face = ribbon_graph.face(label)
    delete_edge(ribbon_graph, label)
    for l in face[1:]:
        contract_edge(ribbon_graph, l)

