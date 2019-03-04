import itertools
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

    def possible_previous_steps(self):
        start_vertex = self.ribbon_graph.vertex(self.start_label)[1:]
        op_labels = [self.ribbon_graph.opposite[l] for l in start_vertex]
        possible_previous_steps = []
        for label in op_labels:
            already_have_vertex = False
            for other_vertex_label in self.ribbon_graph.vertex(label):
                if other_vertex_label in self.labels:
                    already_have_vertex = True
                    break
            if not already_have_vertex:
                possible_previous_steps.append(label)
        return possible_previous_steps

    
    def one_step_continuations(self):
        continuations = []
        for label in self.possible_next_steps():
            new_labels = self.labels[:]
            new_labels.append(label)
            continuations.append(EmbeddedPath(self.ribbon_graph, self.start_label, labels = new_labels) )
        return continuations


    def concatenate(self, other_embedded_path):
        if self.ribbon_graph != other_embedded_path.ribbon_graph:
            raise Exception("To concatenate EmbeddedPaths, must be paths in the same RibbonGraph.")
        if other_embedded_path.start_label in self.possible_next_steps():
            new_labels = self.labels[:]
            new_labels.extend(other_embedded_path.labels)
            return EmbeddedPath(self.ribbon_graph, self.start_label, new_labels)
        else:
            raise Exception("Paths cannot be concatenated.")
    
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


    def reversed(self):
        op_labels = list(reversed([self.ribbon_graph.opposite[l] for l in self.labels]))
        return EmbeddedPath(self.ribbon_graph, op_labels[0], labels = op_labels)
        
class EmbeddedCycle(Path):
    """
    Turn degrees all -1 correspond to faces oriented in the same way as 
    the RibbonGraph (with the face to the left side).
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


    def starting_at(self, new_start_label):
        """
        Start the cycle at new_start_label instead.
        """
        i = self.labels.index(new_start_label)
        new_labels = self.labels[i:-1]
        new_labels.extend(self.labels[:i])
        new_labels.append(new_start_label)
        return EmbeddedCycle(self.ribbon_graph, new_start_label, new_labels)
        
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

    def with_previous_labels(self):
        opposites = [self.ribbon_graph.opposite[l] for l in self.labels]
        opposites.append(opposites.pop(0))
        return zip(self.labels,opposites)

    
    def cut(self):
        R = self.ribbon_graph
        
        for l, o in self.with_previous_labels():
            R = R.vertex_merge_unmerge(l,self.ribbon_graph.next[o])
        return R
        
    def left_side(self):
        R = self.ribbon_graph.copy()
        right_side_labels = [l for labels in self.right_side_labels() for l in labels]
        return R.disconnect_vertices(right_side_labels).restricted_to_connected_component_containing(self.start_label)
        

    def reversed(self):
        op_labels = list(reversed([self.ribbon_graph.opposite[l] for l in self.labels]))
        return EmbeddedCycle(self.ribbon_graph, op_labels[0], labels = op_labels)

    def symmetric_difference(self, other_embedded_path):
        if self.ribbon_graph != other_embedded_path.ribbon_graph:
            raise Exception("To take symmetric difference, both cycles must be in the same ribbon graph.")
        label_set = set(self.labels)
        other_label_set = set(other_embedded_path.labels)
        symmetric_difference = label_set.symmetric_difference(other_label_set)
        for start_label in symmetric_difference:
            break
        return EmbeddedCycle(self.ribbon_graph, start_label, label_set=symmetric_difference)

    def oriented_sum(self, other_embedded_path):
        if self.ribbon_graph != other_embedded_path.ribbon_graph:
            raise Exception("To take symmetric difference, both cycles must be in the same ribbon graph.")
        label_set = set(self.labels)
        other_label_set = set(other_embedded_path.labels)
        new_labels = label_set.union(other_label_set)
        for label in self.labels:
            op_label = self.ribbon_graph.opposite[label]
            if (label in new_labels) and (op_label in new_labels):
                new_labels.remove(label)
                new_labels.remove(op_label)
        for start_label in new_labels:
            break
        return EmbeddedCycle(self.ribbon_graph, start_label, label_set=new_labels)

    def split_at_two_points(self, label1, label2):
        """
        Split into two EmbeddedPaths, one starting at label1, and going
        to the label before label2, and the other starting at label2 and going
        to the label before label1.
        One should be to do path1.concatenate(path2).complete_to_cycle() to 
        return to the original cycle.
        """
        rotated1 = self.starting_at(label1)
        dist_1to2 = rotated1.labels.index(label2)
        labels1 = rotated1.labels[ : dist_1to2 ]

        rotated2 = self.starting_at(label2)
        dist_2to1 = rotated2.labels.index(label1)
        labels2 = rotated2.labels[ : dist_2to1 ]

        return EmbeddedPath(self.ribbon_graph, label1, labels = labels1), EmbeddedPath(self.ribbon_graph, label2, labels = labels2)
    
    def split_along_path(self, splitting_path):
        """
        Given an EmbeddedPath going through starting on the left side of
        the boundary, and ending on the left side of the boundary, use this 
        path to split self into two cycles which each have a portion of self.
        boundary.
        """
        for label1 in splitting_path.possible_next_steps():
            if label1 in self.labels:
                break
        start_vertex = splitting_path.ribbon_graph.vertex(splitting_path.labels[0]) 
        for label2 in start_vertex:
            if label2 in self.labels:
                break

        boundary1, boundary2 = self.split_at_two_points(label1, label2)
        cycle1 = splitting_path.concatenate(boundary1).complete_to_cycle()
        cycle2 = splitting_path.reversed().concatenate(boundary2).complete_to_cycle()

        return cycle1, cycle2
