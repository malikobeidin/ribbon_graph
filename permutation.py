#from sage.all import Permutation as SagePermutation

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
        
            
    def __mul__(self,other_bijection):
        if self.codomain == other_bijection.domain:
            
            return Bijection({label: other_bijection[self[label]] for label in self })
        else:
            raise Exception("Domain/codomain don't match")

    def inverse(self):
        return Bijection({self[label]:label for label in self})
        
class Permutation(Bijection):
    def __init__(self, dictionary={}, cycles = [], verify=True):
        super(Permutation,self).__init__(dictionary=dictionary,verify=False)

        for cycle in cycles:
            self.add_cycle(cycle)

        if verify:
            self.verify()

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
        return self.domain
            
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

    def sage(self):
        labels = list(self.labels())
        cycles = self.cycles()
        i_cycles = [tuple([labels.index(label)+1 for label in cycle]) for cycle in cycles]
        print(i_cycles)
        return SagePermutation(i_cycles)
