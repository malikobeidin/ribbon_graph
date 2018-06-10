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
        if label in self.domain:
            raise Exception("Cannot change function value at {}".format(label))
        if output_label in self.codomain:
            raise Exception("Output {} already in codomain ".format(output_label))
        super(Bijection,self).__setitem__(label,output_label)
        self.domain.add(label)
        self.codomain.add(output_label)

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
            self[cycle[i]] = cycle[(i+1)%len(cycle)]
            Permutation()
        
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
