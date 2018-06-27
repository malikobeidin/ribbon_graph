from ribbon_graph_base import RibbonGraph

class Path(object):
    def __init__(self, ribbon_graph, start_label, labels = [], turn_degrees = []):
        self.ribbon_graph = ribbon_graph
        self.start_label = start_label

        
        if labels:
            if labels[0] != start_label:
                raise Exception("Starting label must be first in list of labels")
            self.turn_degrees = self._compute_turn_degrees_from_labels(labels)
            self.labels = labels
            
        elif turn_degrees:
            self.turn_degrees = turn_degrees
            self.labels = self._compute_labels_from_turn_degrees(turn_degrees)
                        
        else:
            raise Exception("Must specify list of half-edge labels or turn degrees")
        self._make_turn_degrees_positive()

    def _compute_labels_from_turn_degrees(self, turn_degrees):
        labels = [self.start_label]
        label = self.start_label
        for d in turn_degrees:
            label = self.ribbon_graph.opposite[label]
            label = self.ribbon_graph.next.iterate(d, label)
            labels.append(label)
        return labels

    def _make_turn_degrees_positive(self):
        new_turn_degrees = []
        for label, turn_degree in zip(self.labels[1:], self.turn_degrees):
            vertex_valence = len(self.ribbon_graph.vertex(label))
            new_turn_degrees.append( turn_degree % vertex_valence )
        self.turn_degrees = new_turn_degrees
    
    def _compute_turn_degrees_from_labels(self, labels):
        turn_degrees = []
        for i in range(len(labels)-1):
            label, next_label = labels[i], labels[i+1]
            op_label = self.ribbon_graph.opposite[label]
            vertex = self.ribbon_graph.vertex(op_label)
            turn_degrees.append(vertex.index(next_label))
        return turn_degrees


    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,self.labels)

    
    def next_vertex(self):
        return self.ribbon_graph.vertex( self.ribbon_graph.opposite[self.labels[-1]]  )

    def inverse_turn_degrees(self):
        inverse_turn_degrees = []
        opposite = self.ribbon_graph.opposite
        labels = self.labels
        for i in range(len(labels)-1):
            label, next_label = labels[i], labels[i+1]
            op_label = opposite[label]
            vertex = self.ribbon_graph.vertex(op_label)
            inverse_turn_degrees.append(len(vertex) - vertex.index(next_label))
        return inverse_turn_degrees
        
    
class EmbeddedPath(Path):
    def __init__(self, ribbon_graph, start_label, labels = [], turn_degrees = [], label_set = set([])):
        if labels or turn_degrees:
            
            super(EmbeddedPath,self).__init__(ribbon_graph,
                                              start_label,
                                              labels=labels,
                                              turn_degrees = turn_degrees)
        elif label_set:
            self.ribbon_graph = ribbon_graph
            self.start_label = start_label
            
            labels = self._compute_labels_from_label_set(label_set)
            super(EmbeddedPath,self).__init__(ribbon_graph,
                                              start_label,
                                              labels=labels,
                                              turn_degrees = [])
        else:

            raise Exception("Must specify either labels, turn degrees, or the set of labels in the embedded path.")

        self._verify_embedded()

    def _compute_labels_from_label_set(self, label_set):
        labels = []
        label = self.start_label
        while label_set:
            labels.append(label)
            label = self.ribbon_graph.opposite[label]
            vertex = self.ribbon_graph.vertex(label)
            possible_next_labels = [l for l in label_set if l in vertex]
            if len(possible_next_labels) != 1:
                raise Exception("Label set does not define path")
            label = possible_next_labels[0]
            label_set.remove(label)
        return labels
            
    def _verify_embedded(self):
        vertices = [frozenset(self.ribbon_graph.vertex(label)) for label in self.labels]
        if len(set(vertices)) < len(vertices):
            raise Exception("Path is not embedded")

    def possible_next_steps(self):
        next_vertex = self.next_vertex()
        for label in next_vertex:
            if label in self.labels:
                return []
        return next_vertex[1:]
        
    def one_step_continuations(self):
        continuations = []
        for label in self.possible_next_steps():
            new_labels = self.labels[:]
            new_labels.append(label)
            continuations.append(EmbeddedPath(self.ribbon_graph, self.start_label, labels = new_labels) )
        return continuations

    
    def is_completable_to_cycle(self):
        return self.start_label in self.next_vertex()
    
    def complete_to_cycle(self):
        if self.is_completable_to_cycle():
            new_labels = self.labels[:]
            new_labels.append(self.start_label)
            return EmbeddedCycle(self.ribbon_graph, self.start_label, labels=new_labels)
        else:
            raise Exception("Not completable to a cycle")

    def __len__(self):
        return len(self.labels)

        
        
class EmbeddedCycle(Path):
    """
    Turn degrees all -1 correspond to faces
    """
    def __init__(self, ribbon_graph, start_label, labels = [], turn_degrees = [], label_set = set([])):
        if labels or turn_degrees:
            
            super(EmbeddedCycle,self).__init__(ribbon_graph,
                                              start_label,
                                              labels=labels,
                                              turn_degrees = turn_degrees)
        elif label_set:
            self.ribbon_graph = ribbon_graph
            self.start_label = start_label
            labels = self._compute_labels_from_label_set(label_set)
            super(EmbeddedCycle,self).__init__(ribbon_graph,
                                              start_label,
                                              labels=labels,
                                              turn_degrees = [])
        else:

            raise Exception("Must specify either labels, turn degrees, or the set of labels in the embedded cycle.")

        self._verify_embedded_up_to_final_label()
        self._verify_cycle()

    def _compute_labels_from_label_set(self, label_set):
        labels = []
        label = self.start_label
        while label_set:
            labels.append(label)
            label = self.ribbon_graph.opposite[label]
            vertex = self.ribbon_graph.vertex(label)
            possible_next_labels = [l for l in label_set if l in vertex]
            if len(possible_next_labels) != 1:
                raise Exception("Label set does not define path")
            label = possible_next_labels[0]
            label_set.remove(label)
        labels.append(self.start_label)
        return labels
            
    def _verify_embedded_up_to_final_label(self):
        vertices = [frozenset(self.ribbon_graph.vertex(label)) for label in self.labels[:-1]]
        if len(set(vertices)) < len(vertices):
            raise Exception("Cycle is not embedded")

    def _verify_cycle(self):
        if self.labels[-1] != self.start_label:
            raise Exception("Not a cycle")

    def __len__(self):
        return len(self.labels)-1

        
    def left_side_labels(self):
        left_sides = []
        next_inv = self.ribbon_graph.next.inverse()
        inv_turn_degrees = self.inverse_turn_degrees()
        for label, turn_degree in zip(self.labels[:-1],inv_turn_degrees):
            next_label = self.ribbon_graph.opposite[label]
            left_side_labels = []
            
            for j in range(turn_degree-1):
                next_label = next_inv[next_label]
                left_side_labels.append(next_label)
            left_sides.append(left_side_labels)
        return left_sides

    
        
    def right_side_labels(self):
        right_sides = []
        for label, turn_degree in zip(self.labels[:-1],self.turn_degrees):
            next_label = self.ribbon_graph.opposite[label]
            right_side_labels = []
            for j in range(turn_degree-1):
                next_label = self.ribbon_graph.next[next_label]
                right_side_labels.append(next_label)
            right_sides.append(right_side_labels)
        return right_sides

    
    def left_side(self):
        R = self.ribbon_graph.copy()
        right_side_labels = [l for labels in self.right_side_labels() for l in labels]
        return R.disconnect_vertices(right_side_labels).restricted_to_connected_component_containing(self.start_label)
        
