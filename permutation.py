from sage.all import Permutation as SagePermutation

class Bijection(dict):
    def __init__(self, dictionary, verify=True):
        if verify:
            assert len(set(self.keys())) == len(set(self.values()))
        

class Permutation:
    def __init__(self, perm_dict={}, cycles = [], verify=True):
        self.perm_dict = perm_dict
        if verify:
            self.verify()
        
        if cycles:
            if perm_dict:
                raise Exception("Can't have both perm_dict and cycles")
            else:
                permutations = [Permutation(self._cycle_permutation(cycle)) for cycle in cycles]
                perm = Permutation()
                for cycle_perm in permutations:
                    perm = perm*cycle_perm
                self.perm_dict=perm.perm_dict
                            

    def verify(self):
        assert set(self.perm_dict.keys()) == set(self.perm_dict.values())

    def __getitem__(self, label):
        if label in self.perm_dict:
            return self.perm_dict[label]
        else:
            return label

    def __len__(self):
        return len(self.perm_dict)

    def __contains__(self, label):
        return label in self.perm_dict

    def  __iter__(self):
        return self.perm_dict.__iter__()

    def __mul__(self, other_permutation):
        new_labels = self.labels().union(other_permutation.labels())
        new_perm_dict = {label: other_permutation[self[label]] for label in new_labels}
        return Permutation(new_perm_dict)
        
    def delete_fixed_points(self):
        return Permutation({label: self[label] for label in self if label!=self[label]})
        
    def labels(self):
        return set(self.perm_dict.keys())

    def _cycle_permutation(self,cycle):
        cycle_permutation = {}
        for i, label in enumerate(cycle):
            cycle_permutation[label] = cycle[(i+1)%len(cycle)]
        return cycle_permutation
    
    def add_cycle(self, cycle):
        #Not very smart way of doing this
        new_perm_dict = {i:cycle_permutation[self[i]] for i in self}
        self.perm_dict = new_perm_dict
        
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
        inverse_perm_dict = {self[label]:label for label in self}
        return Permutation(inverse_perm_dict)
    
    def __repr__(self):
        return str(self.perm_dict)

    def sage(self):
        labels = list(self.labels())
        cycles = self.cycles()
        i_cycles = [tuple([labels.index(label)+1 for label in cycle]) for cycle in cycles]
        print(i_cycles)
        return SagePermutation(i_cycles)
