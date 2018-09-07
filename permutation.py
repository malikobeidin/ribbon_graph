#from sage.all import Permutation as SagePermutation
from random import shuffle
class Bijection(dict):
    def __init__(self, dictionary={}, verify=True):
        self.domain = set()
        self.codomain = set()

        for label in dictionary:
            self[label] = dictionary[label]
        if verify:
            assert len(self.domain) == len(self.codomain)

    def __setitem__(self, label, output_label):
#        if label in self.domain:
#            raise Exception("Cannot change function value at {}".format(label))
#        if output_label in self.codomain:
#            raise Exception("Output {} already in codomain ".format(output_label))
        super(Bijection,self).__setitem__(label,output_label)
        self.domain.add(label)
        self.codomain.add(output_label)

    def add_label(self, label, output_label):
        if label in self.domain:
            raise Exception("Cannot change function value at {}".format(label))
        if output_label in self.codomain:
            raise Exception("Output {} already in codomain ".format(output_label))
        super(Bijection,self).__setitem__(label,output_label)
        self.domain.add(label)
        self.codomain.add(output_label)

    def remove_label(self, label):
        output = self.pop(label)
        self.domain.remove(label)
        self.codomain.remove(output)
        
        
    def __repr__(self):
        return ''.join(['{}->{}\n'.format(label,self[label]) for label in self])


    
    def act(self, label):
        if label not in self:
            return label
        else:
            return self[label]
            
    def composed_with(self,other_bijection):
        if self.codomain == other_bijection.domain:
            
            return Bijection({label: other_bijection[self[label]] for label in self })
        else:
            raise Exception("Domain/codomain don't match")

    def inverse(self):
        return Bijection({self[label]:label for label in self})

    def restricted_to(self, labels):
        return Bijection({label: self[label] for label in labels})

    
        
class Permutation(Bijection):
    def __init__(self, dictionary={}, cycles = [], verify=True):
        super(Permutation,self).__init__(dictionary=dictionary,verify=False)

        for cycle in cycles:
            self.add_cycle(cycle)

        if verify:
            self.verify()

    def __mul__(self,other_permutation):
        combined_labels = self.labels().union(other_permutation.labels())
        return Permutation({label: other_permutation.act(self.act(label)) for label in combined_labels })


    def add_cycle(self,cycle):
        for i in range(len(cycle)):
            self.add_label(cycle[i], cycle[(i+1)%len(cycle)])
#            self[cycle[i]] = cycle[(i+1)%len(cycle)]
        
    def verify(self):
        assert self.domain == self.codomain
        
    def fixed_points_removed(self):
        return Permutation({label: self[label] for label in self if label!=self[label]})

    def fixed_points(self):
        return set(i for i in self if self[i]==i)
    
    def labels(self):
        return set(self.domain)
            
    def cycle(self, label):
        c = [label]
        next_label = self[label]
        while next_label != label:
            c.append(next_label)
            next_label = self[next_label]
        return c
            
    def cycles(self):
        labels = self.labels()
        cycles = []
        while labels:
            label = labels.pop()
            cycle = self.cycle(label)
            cycles.append(cycle)
            labels = labels-set(cycle)
        return cycles

    def inverse(self):
        return Permutation({self[label]:label for label in self})

    def restricted_to(self, labels):
        for label in labels:
            assert self[label] in labels
        return Permutation({label: self[label] for label in labels})
        
    
    def relabeled(self, bijection):
        return Permutation(bijection.inverse().composed_with(self).composed_with(bijection))

    def relabel_with_integers(self):
        relabeling = Bijection({l:i for i,l in enumerate(self.labels())})
        return self.relabeled(relabeling), relabeling
    
    def sage(self):
        labels = list(self.labels())
        cycles = self.cycles()
        i_cycles = [tuple([labels.index(label)+1 for label in cycle]) for cycle in cycles]
        print(i_cycles)
        return SagePermutation(i_cycles)
    
    def iterate(self, n, label):
        if n<0:
            inverse = self.inverse()
            for i in range(abs(n)):
                label = inverse[label]
            return label
        elif n>0:
            for i in range(n):
                label = self[label]
            return label
        else:
            return label

    def undo_two_cycle(self, label):
        """
        If label is in a two-cycle, then force label and self[label] to be
        fixed points of self. This will correspond to cutting an edge in a 
        RibbonGraph. Note that this does not make a new permutation object,
        it alters the internal data of self.
        """
        cycle = self.cycle(label)
        if len(cycle) == 2:
            for l in cycle:
                self[l] = l 
        else:
            raise Exception("Given label not in 2-cycle")
        
    def insert_between(self, new_label, previous_label):
        """
        Insert new_label into the cycle containing previous_label, between 
        previous_label and self[previous_label]. That is, change

        previous_label --> self[previous_label]
        
        to
        
        previous_label --> new_label --> self[previous_label]

        This function does not make a new Permutation; it alters self.
        new_label must not be already in the permutation.
        """
        if new_label in self.labels():
            raise Exception("Cannot insert label because it is already used in the permutation")
        self[new_label] = self[previous_label]
        self[previous_label] = new_label
        
    def split_cycle_at(self, label1, label2):
        """
        Takes two labels on the same cycle and splits the cycle into two.
        It short-circuits the cycle after label1 to skip to the label after
        label2, and vice versa. Here's an example:

        sage: P = Permutation(cycles = [(1,2,3,4,5)])
        sage: P.split_cycle_at(3,5)
        sage: P.cycles()
        [[1, 2, 3], [4, 5]]

        This will correspond to splitting a vertex for ribbon graphs. This
        function doesn't make a new permutation, it alters self.
        """
        if label2 not in self.cycle(label1):
            raise Exception("The two labels are not on the same cycle.")
        label1_next = self[label1]
        label2_next = self[label2]

        self[label1] = label2_next
        self[label2] = label1_next

    def merge_cycles_at(self, label1, label2):
        """
        Takes two labels on different cycles and merge the cycles into one.
        It short-circuits the cycle after label1 to skip to the label after
        label2, and vice versa. Here's an example:

        sage: P = Permutation(cycles = [(1,2,3,4,5)])
        sage: P.split_cycle_at(3,5)
        sage: P.cycles()
        [[1, 2, 3], [4, 5]]

        This will correspond to merging a vertex for ribbon graphs. This
        function doesn't make a new permutation, it alters self.
        """
        if label2 in self.cycle(label1):
            raise Exception("The two labels are on the same cycle.")
        label1_next = self[label1]
        label2_next = self[label2]

        self[label1] = label2_next
        self[label2] = label1_next

    def remove_cycle(self, label):
        for l in self.cycle(label):
            self.remove_label(l)
        
    def union(self, other_permutation):
        U = Permutation()
        for i in self:
            U[i] = self[i]
        for i in other_permutation:
            U[i] = other_permutation[i]
        return U
        
    def disjoint_union(self, other_permutation):
        combined = {}
        for label in self:
            combined[(label,0)] = (self[label],0)
        for label in other_permutation:
            combined[(label,1)] = (other_permutation[label],1)
        return Permutation(combined)
        
        
def permutation_from_bijections(bijections):
    B = Bijection()
    for bijection in bijections:
        for key in bijection:
            B[key] = bijection[key]
    return Permutation(B)

def random_permutation(labels):
    permuted_labels = list(labels)
    shuffle(permuted_labels)
    return Permutation({l1: l2 for l1,l2 in zip(labels,permuted_labels)})

def random_cycle(labels):
    shuffle(labels)
    return Permutation(cycles=[labels])

def four_cycles(num_vertices):
    return Permutation(cycles=[[4*i,4*i+1,4*i+2,4*i+3] for i in range(num_vertices)])
