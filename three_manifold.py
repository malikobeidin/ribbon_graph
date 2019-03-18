from permutation import Permutation, Bijection
from ribbon_graph_base import *
from local_moves import contract_edge

class Triangulation(object):
    def __init__(self, tetrahedra_ribbon_graph, glued_to, check_consistency = True):
        """
        Specify a triangulation of a 3-manifold with:
               tetrahedra_ribbon_graph: a ribbon graph whose faces are all 
                      size 3, and whose connected components are all
                      the tetrahedral map.
               glued_to: a permutation which shows which faces are glued. 
                      A directed edge in a tetrahedron uniquely specifies
                      how to glue two faces, by gluing the faces to the left
                      of each directed edge. This permutation must commute with
                      the next_corner permutation of the tetrahedra_ribbon_graph
                      to make it so that faces are glued consistently.
        """
        self.tetrahedra_ribbon_graph = tetrahedra_ribbon_graph
        if check_consistency:            
            next_corner = self.tetrahedra_ribbon_graph.next_corner()
            glued_to = next_corner.make_commute_along_cycles(glued_to)
        self.glued_to = glued_to

    def tetrahedra(self):
        return self.tetrahedra_ribbon_graph.connected_components()

    def edges(self):
        return (self.tetrahedra_ribbon_graph.next * D.glued_to).cycles()


        
    
    def vertex_links(self):
        return RibbonGraph([self.glued_to, self.tetrahedra_ribbon_graph.next.inverse()])

class TruncatedTetrahedron(object):
    def __init__(self, tet):
        self.tetrahedron = tet
        self.ribbon_graph = truncate_vertices(tetrahedron().dual())
        self.pairing = self.snappy_tetrahedron_to_edge_pairing()
        self.labeled = self.labeled_ribbon_graph()
        
    def vertex(self, directed_edge):
        """
        From a directed edge 'ab', get the vertex in the truncated tetrahedron
        which has this vertex. Each directed edge in the original tetrahedron
        corresponds to a vertex in the truncated tetrahedron.
        """
        return self.ribbon_graph.vertex(directed_edge)
    
    def boundary_edge_from_directed_edge(self, directed_edge):
        """
        Return a label on the triangular face on which the vertex corresponding
        to directed_edge is located.
        """
        return self.ribbon_graph.next[directed_edge]

    def boundary_triangle_from_directed_edge(self, directed_edge):
        return self.ribbon_graph.face(self.ribbon_graph.next[directed_edge])

    def snappy_tetrahedron_to_edge_pairing(self):
        tet = self.tetrahedron
        gluing = tet.Gluing
        neighbor = tet.Neighbor

        directed_edge_choices = ['32','23','10','01']
        corresponding_left_faces = [7, 11, 13, 14] #[F3, F2, F1, F0]

        permutations = [Permutation(gluing[i].dict) for i in corresponding_left_faces]
        corresponding_neighbors = [neighbor[i].Index for i in corresponding_left_faces]
        pairing = []
        for e, perm, index in zip(directed_edge_choices,permutations, corresponding_neighbors):
            print(tet.Index, index)
            print(perm)
            vs = self.boundary_edge_from_directed_edge(e)
            new_e = ''.join(reversed([str(perm[int(v)]) for v in e]))
            new_vs = self.boundary_edge_from_directed_edge(new_e)
#            new_vs = ''.join([new_vs[2], new_vs[3], new_vs[0], new_vs[1]])

            print(e,new_e)
            print(vs, new_vs)
            print(self.ribbon_graph.face(vs), self.ribbon_graph.face(new_vs))
            pairing.append( (str(tet.Index)+vs, str(index)+new_vs) )
            print('\n\n')
        return pairing

    def labeled_ribbon_graph(self):
        l = str(self.tetrahedron.Index)
        new_opposite = {(l+i):(l+self.ribbon_graph.opposite[i]) for i in self.ribbon_graph.opposite}
        new_next = {(l+i):(l+self.ribbon_graph.next[i]) for i in self.ribbon_graph.next}
        
        return RibbonGraph([Permutation(new_opposite), Permutation(new_next)])



def heegaard_surface(mcomplex):
    truncated_tets = [TruncatedTetrahedron(tet) for tet in mcomplex.Tetrahedra]
    truncated_tet = truncated_tets.pop()
    R = truncated_tet.labeled
    pairings = [truncated_tet.pairing]
    for truncated_tet in truncated_tets:
        R = R.union(truncated_tet.labeled)
        pairings.append(truncated_tet.pairing)
    print(pairings)
    all_face_labels = []
    for pairing in pairings:
        for label1, label2 in pairing:
            all_face_labels.append((R.face(label1), R.face(label2)))
    print(all_face_labels)
    for pairing in pairings:
        for label1, label2 in pairing:
            labels = R.labels()
            if (label1 in labels) and (label2 in labels):            
                R = R.glue_faces(label1, label2)
    return R
    


N   = 0  # 0000
V0  = 1  # 0001
V1  = 2  # 0010
E01 = 3  # 0011 <-----|
V2  = 4  # 0100       |
E02 = 5  # 0101 <---| |
E21 = 6  # 0110 <-| | |
F3  = 7  # 0111   | | |
V3  = 8  # 1000   | | |  Opposite edges
E03 = 9  # 1001 <-| | |
E13 = 10 # 1010 <---| |
F2  = 11 # 1011       |
E32 = 12 # 1100 <-----|
F1  = 13 # 1101
F0  = 14 # 1110
T   = 15 # 1111

# User-friendly?

E10 = 3
E20 = 5
E12 = 6
E30 = 9
E31 = 10
E23 = 12


# A simplex is oriented like this:  
#     1     
#    /|\    
#   / | \   
#  /  |  \  
# 2---|---3 
#  \  |  /  
#   \ | /   
#    \|/    
#     0
#

def tetrahedron(label=None):
    next = Permutation(cycles=[('01','02','03'),
                               ('12','10','13'),
                               ('20','21','23'),
                               ('30','32','31')])
    opposite = Permutation(cycles=[('01','10'),
                                   ('02','20'),
                                   ('03','30'),
                                   ('12','21'),
                                   ('13','31'),
                                   ('23','32')])
    if label:
        return RibbonGraph([opposite.append_label(label), next.append_label(label)])
    else:
        return RibbonGraph([opposite, next])
    
def tetrahedron_old(label=None):
    next = Permutation(cycles=[((0,1),(0,3),(0,2)),
                               ((1,2),(1,3),(1,0)),
                               ((2,0),(2,3),(2,1)),
                               ((3,0),(3,1),(3,2))])
    opposite = Permutation(cycles=[((0,1),(1,0)),
                                   ((0,2),(2,0)),
                                   ((0,3),(3,0)),
                                   ((1,2),(2,1)),
                                   ((1,3),(3,1)),
                                   ((2,3),(3,2))])
    if label:
        return RibbonGraph([opposite.append_label(label), next.append_label(label)])
    else:
        return RibbonGraph([opposite, next])

def snappy_tetrahedron_to_face_gluings(snappy_tetrahedron):
    directed_edge_choices = ['01','10','23','32']
    corresponding_left_faces = [7, 11, 13, 14] #[F3, F2, F1, F0]
    permutations = [Permutation(snappy_tetrahedron.Gluing[i].dict) for i in corresponding_left_faces]
    neighbor_indices = [snappy_tetrahedron.Neighbor[i].Index for i in corresponding_left_faces]
    T = tetrahedron()
    face_pairing = []
    edge_pairing = []
    face_label_pairing = []
    i = snappy_tetrahedron.Index
    
    for directed_edge, permutation, neighbor_i in zip(directed_edge_choices, permutations, neighbor_indices):

        
        directed_edge_permuted = ''.join(reversed([str(permutation[int(s)]) for s in directed_edge]))
        edge_pairing.append([(i,directed_edge), (neighbor_i,directed_edge_permuted)])
        face_pairing.append([(i,T.face(directed_edge)), (neighbor_i,T.face(directed_edge_permuted))])
        missing_label = set('0123')-set(T.face(directed_edge)[0])-set(T.face(directed_edge)[1])-set(T.face(directed_edge)[2])
        missing_label = missing_label.pop()
        missing_neighbor_label = set('0123')-set(T.face(directed_edge_permuted)[0])-set(T.face(directed_edge_permuted)[1])-set(T.face(directed_edge_permuted)[2])
        missing_neighbor_label = missing_neighbor_label.pop()
        face_label_pairing.append(['F{} of tet{}'.format(missing_label, i),'F{} of tet{}'.format(missing_neighbor_label, neighbor_i)])

    return  face_label_pairing

class Tetrahedron(object):
    def __init__(self, snappy_tetrahedron):
        self.ribbon_graph = tetrahedron()
        self.cut_ribbon_graph = truncate_vertices(thicken_edges(tetrahedron()))
        self.snappy_tetrahedron = snappy_tetrahedron
        self.snappy_label_to_code = {0:14, 1:13, 2:11, 3:7}

    def face_from_missing_vertex(self, v):
        if v == 0:
            return self.ribbon_graph.face('32')
        elif v == 1:
            return self.ribbon_graph.face('23')
        elif v == 2:
            return self.ribbon_graph.face('10')
        elif v == 3:
            return self.ribbon_graph.face('01')
        else:
            raise Exception()

    def face_from_snappy_label(self, i):
        if i == 14:
            return self.ribbon_graph.face('32')
        elif i == 13:
            return self.ribbon_graph.face('23')
        elif i == 11:
            return self.ribbon_graph.face('10')
        elif i == 7:
            return self.ribbon_graph.face('01')
        else:
            raise Exception()

    def edge_mapping(self):
        mappings = []
        tet_index = str(self.snappy_tetrahedron.Index)
        for i in [14,13,11,7]:
            assert self.snappy_tetrahedron.Gluing[i].sign() == 1
            perm = Permutation(self.snappy_tetrahedron.Gluing[i].dict)
            neighbor = str(self.snappy_tetrahedron.Neighbor[i].Index)
            face = self.face_from_snappy_label(i)
            edge_mapping = {}
            for directed_edge in face:
                directed_edge_permuted = ''.join([str(perm[int(s)]) for s in directed_edge])
                directed_edge_permuted = self.ribbon_graph.opposite[directed_edge_permuted]
                edge_mapping[tet_index+'|'+directed_edge]=neighbor+'|'+directed_edge_permuted
            mappings.append(edge_mapping)
        return mappings

    def face_pairing(self):

        face_mapping = {}
        tet_index = str(self.snappy_tetrahedron.Index)
        for i in range(4):
            code = self.snappy_label_to_code[i]
            assert self.snappy_tetrahedron.Gluing[code].sign() == 1
            perm = Permutation(self.snappy_tetrahedron.Gluing[code].dict)
            neighbor = str(self.snappy_tetrahedron.Neighbor[code].Index)
            directed_edge = self.face_from_missing_vertex(i)[0]
            directed_edge_permuted = ''.join([str(perm[int(s)]) for s in directed_edge])
            directed_edge_permuted = self.ribbon_graph.opposite[directed_edge_permuted]
            opposite_face = self.snappy_label_from_face(directed_edge_permuted)
            face_mapping['F{} of tet{}'.format(i,tet_index)]='F{} of tet{}'.format(opposite_face,neighbor)

        return face_mapping

    def snappy_label_from_face(self, directed_edge):
        face = self.ribbon_graph.face(directed_edge)
        v = set('0123')
        for l in face:
            v = v-set(l)
        return v.pop()

    def directed_edge_to_cut_face_label(self, directed_edge):
        return directed_edge+'_1'

    def edge_mapping_on_cut_faces(self):
        edge_mapping = self.edge_mapping()
        return [{i+'_1':j+'_1' for i,j in mapping.items()} for mapping in edge_mapping]


    def with_tet_label(self):
        si = str(self.snappy_tetrahedron.Index)+'|'
        op = self.cut_ribbon_graph.opposite
        new_op = Permutation({si+label : si+op[label] for label in op})
        next = self.cut_ribbon_graph.next
        new_next = Permutation({si+label : si+next[label] for label in next})
        return RibbonGraph([new_op,new_next])
    
class TriangulationSkeleton(object):
    def __init__(self, mcomplex):
        self.tetrahedra = [Tetrahedron(tet) for tet in mcomplex.Tetrahedra]

        self.ribbon_graph = RibbonGraph([Permutation(), Permutation()])
        for tet in self.tetrahedra:
            self.ribbon_graph = self.ribbon_graph.union(tet.with_tet_label())
        self.glue_boundary_faces()
        self.classify_lace_components()

    def pair_edge_mappings(self):
        edge_mappings = []
        paired =  []
        for tet in self.tetrahedra:
            edge_mappings.extend(tet.edge_mapping_on_cut_faces())
        while edge_mappings:
            mapping = edge_mappings.pop()
            source, target = mapping.items()[0]
            for other_mapping in edge_mappings:
                if target in other_mapping:
                    edge_mappings.remove(other_mapping)
                    paired.append((mapping, other_mapping))

        return paired

    def glue_boundary_faces(self):
        paired = self.pair_edge_mappings()
        self.cycles = []
        for mapping, other_mapping in paired:
            for label in mapping:
                assert other_mapping[mapping[label]] == label
            label1, label2 = mapping.popitem()
            #print(label1, label2)
            self.cycles.append(self.ribbon_graph.face(label1))
            self.ribbon_graph = self.ribbon_graph.glue_faces(label1,label2)
        

    def classify_lace_components(self):
        lace_components = set(map(tuple, self.ribbon_graph.lace_components()))
        self.face_curves = []
        self.edge_curves = []

        while lace_components:
            lc = lace_components.pop()
            op_lc = None
            for other_lc in lace_components:
                if self.ribbon_graph.opposite[lc[0]] in other_lc:
                    op_lc = other_lc
                    break
            lace_components.remove(op_lc)
            is_face_curve = False
            for cycle in self.cycles:
                if (lc[0] in cycle) or (op_lc[0] in cycle):
                    is_face_curve = True
                    break
            if is_face_curve:
                self.face_curves.append( (lc, op_lc) )
            else:
                self.edge_curves.append( (lc, op_lc) )

    
        next_corner = self.ribbon_graph.next_corner()

        edge_curve_set = set(self.edge_curves)
        self.edge_curves = []
        while edge_curve_set:
            ec,ec_op = edge_curve_set.pop()
            opposite_side_label1 = next_corner[next_corner[ec[0]]]
            opposite_side_label2 = next_corner[next_corner[ec_op[0]]]

            for other_ec, other_ec_op in edge_curve_set:                
                if (opposite_side_label1 in other_ec) or (opposite_side_label1 in other_ec_op) or (opposite_side_label2 in other_ec) or (opposite_side_label2 in other_ec_op):
                    edge_curve_set.remove((other_ec,other_ec_op))
                    self.edge_curves.append((ec, ec_op, other_ec, other_ec_op))
                    break

    def print_data(self):
        label_numbering = {}
        R = self.ribbon_graph
        labels = R.labels()
        i = 1
        X = R.euler_characteristic()
        assert X % 2 == 0
        
        print('Genus: {}'.format((2-X)//2))
        while labels:            
            l = labels.pop()
            o = R.opposite[l]
            labels.remove(o)
            label_numbering[l] = i
            label_numbering[o] = -i
            i+=1
        print('vertices:')
        for v in R.vertices():
            print([label_numbering[l] for l in v])
        print('faces:')
        for f in R.faces():
            print([label_numbering[l] for l in f])
        print('face curves:')
        for fc, op_fc in self.face_curves:
            print([label_numbering[l] for l in fc])
            print([label_numbering[l] for l in op_fc])
            print('')
        print('edge curves:')
        for ec, op_ec, adj_ec, op_adj_ec in self.edge_curves:
            print([label_numbering[l] for l in ec])
            print([label_numbering[l] for l in op_ec])
            print([label_numbering[l] for l in adj_ec])
            print([label_numbering[l] for l in op_adj_ec])
            print('')


    def collapse_to_single_vertex(self):
        found_non_loop = True
        R = self.ribbon_graph.copy()
        while found_non_loop:
            found_non_loop = False
            for label in R.labels():
                if R.opposite[label] not in R.vertex(label):
                    contract_edge(R, label)
                    found_non_loop = True
                    break
        return R
            
def truncate_vertices(ribbon_graph):
    new_opposite = dict(ribbon_graph.opposite)
    next_inverse = ribbon_graph.next.inverse()
    new_next = {}
    for label in ribbon_graph.labels():
        next_label = ribbon_graph.next[label]
        previous_label = next_inverse[label]
        new_next[label]= label+','+next_label
        new_next[label+','+previous_label]= label
        new_next[label+','+next_label]= label+','+previous_label
        new_opposite[label+','+next_label]=next_label+','+label
        new_opposite[next_label+','+label]=label+','+ next_label

    new_opposite  = Permutation(new_opposite)
    new_next = Permutation(new_next)
    return RibbonGraph(permutations=[new_opposite,new_next])


def thicken_edges(ribbon_graph):
    new_op = {}
    for label in ribbon_graph.labels():
        old_op_label = ribbon_graph.opposite[label]
        new_op[label+'_0']=old_op_label+'_1'
        new_op[label+'_1']=old_op_label+'_0'
    new_next = {}
    for label in ribbon_graph.labels():
        old_next_label = ribbon_graph.next[label]
        new_next[label+'_0']=label+'_1'
        new_next[label+'_1']=old_next_label+'_0'

    new_op  = Permutation(new_op)
    new_next = Permutation(new_next)

    return RibbonGraph(permutations=[new_op,new_next])
    
def truncate_vertices_old(ribbon_graph):
    new_opposite = dict(ribbon_graph.opposite)
    next_inverse = ribbon_graph.next.inverse()
    new_next = {}
    for label in ribbon_graph.labels():
        next_label = ribbon_graph.next[label]
        previous_label = next_inverse[label]
        new_next[label]= (label, next_label)
        new_next[(label,previous_label)]= label
        new_next[(label,next_label)]= (label,previous_label)
        new_opposite[(label,next_label)]=(next_label,label)
        new_opposite[(next_label,label)]=(label, next_label)

    new_opposite  = Permutation(new_opposite)
    new_next = Permutation(new_next)
    return RibbonGraph(permutations=[new_opposite,new_next])


def triangulation_from_pairing(pairing):
    """
    Given pairs of tuples ((directed_edge1,tet_label1),(directed_edge2,tet_label2)), return a Triangulation object with those gluings.
    """
    gluing = Permutation(cycles=pairing)
    tetrahedra = {}
    for pair1, pair2 in pairing:
        directed_edge1, tet_label1 = pair1
        directed_edge2, tet_label2 = pair2

        if tet_label1 not in tetrahedra:
            tetrahedra[tet_label1] = tetrahedron(tet_label1)
        if tet_label2 not in tetrahedra:
            tetrahedra[tet_label2] = tetrahedron(tet_label2)

    U = RibbonGraph([Permutation(), Permutation()])
    for T in tetrahedra.values():
        U = U.union(T)
    return Triangulation(U, gluing)

def triangulation_from_tuples(tuples):
    """
    Given 6-tuples (v1, v2 ,tet_label1, w1, w2 , tet_label2), return a Triangulation object with those gluings.
    """

    tetrahedra = {}
    parsed = []
    for six_tuple in tuples:
        v1, v2 , tet_label1, w1, w2, tet_label2 = six_tuple

        if tet_label1 not in tetrahedra:
            tetrahedra[tet_label1] = tetrahedron(tet_label1)
        if tet_label2 not in tetrahedra:
            tetrahedra[tet_label2] = tetrahedron(tet_label2)

        parsed.append( (((v1,v2),tet_label1), ((w1,w2),tet_label2) ))
    U = RibbonGraph([Permutation(), Permutation()])
    for T in tetrahedra.values():
        U = U.union(T)

    gluing = Permutation(cycles=parsed)
    return Triangulation(U, gluing)





def doubled():
    return [(0,1,0,0,1,1),
            (1,2,0,1,2,1),
            (2,0,0,2,0,1),
            (1,0,0,1,0,1)]


def doubled2():
    return [(0,1,0,1,0,1),
            (1,2,0,2,1,1),
            (2,0,0,0,2,1),
            (1,0,0,0,1,1)]

import snappy
def test_one_skeleton(limit):
    i = 0
    for M in snappy.OrientableClosedCensus:
        if i > limit:
            break
        i += 1
        MC = snappy.snap.t3mlite.Mcomplex(M)
        S = TriangulationSkeleton(snappy.snap.t3mlite.Mcomplex(M))
        ec = S.ribbon_graph.euler_characteristic()
        cc = len(S.ribbon_graph.connected_components())
        lc = len(S.ribbon_graph.lace_components())
        
#        print(lc == (sum(e.valence() for e in MC.Edges)+sum(1 for f in MC.Faces)))
        print(lc == 4*len(MC.Edges)+2*len(MC.Faces))
        print([len(v) == 4 for v in S.ribbon_graph.vertices()])
        if (ec >= 0) or (ec%2 != 0) or (cc>1):
            print(M)
            

def filled_triangulation_and_triangulation_skeleton(snappy_string):
    M = snappy.Manifold(snappy_string)
    M.dehn_fill((1,0))
    MF = M.filled_triangulation()
    MC = snappy.snap.t3mlite.Mcomplex(MF)
    return MC, TriangulationSkeleton(MC)
