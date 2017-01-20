#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module contains classes for simulation and inference for a binary branching process with mutation
in which the tree is collapsed to nodes that count the number of clonal leaves of each type
'''

from __future__ import division, print_function
import scipy, warnings, random, cPickle
from scipy.misc import logsumexp
from scipy.optimize import minimize, check_grad

import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt
from matplotlib import rc, ticker
import pandas as pd
import seaborn as sns
from scipy.stats import probplot
from ete3 import TreeNode, NodeStyle, TreeStyle, TextFace, add_face_to_node, CircleFace, faces, AttrFace
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna

scipy.seterr(all='raise')

class LeavesAndClades():
    '''
    This is a base class for simulating, and computing likelihood for, an infinite type branching
    process with branching probability p, mutation probability q, and we collapse mutant clades off the
    root type and consider just the number of clone leaves, c, and mutant clades, m.

      /\
     /\ ^          (3)
      /\     ==>   / \\
       /\\
        ^
    '''
    def __init__(self, params=None, c=None, m=None):
        '''initialize with branching probability p and mutation probability q, both in the unit interval'''
        if params is not None:
            p, q = params
            if not (0 <= p <= 1 and 0 <= q <= 1):
                raise ValueError('p and q must be in the unit interval')
        self._nparams = 2#len(params)
        self.params = params
        if c is not None or m is not None:
            if not (c >= 0) and (m >= 0) and (c+m > 0):
                raise ValueError('c and m must be nonnegative integers summing greater than zero')
            self.c = c
            self.m = m

    def simulate(self):
        '''simulate the number of clone leaves and mutant clades off a root node'''
        if self.params[0]>=.5:
            warnings.warn('p >= .5 is not subcritical, tree simulations not garanteed to terminate')
        if self.params is None:
            raise ValueError('params must be defined for simulation\n')

        # let's track the tree in breadth first order, listing number clone and mutant descendants of each node
        # mutant clades terminate in this view
        cumsum_clones = 0
        len_tree = 0
        self.c = 0
        self.m = 0
        # while termination condition not met
        while cumsum_clones > len_tree - 1:
            if random.random() < self.params[0]:
                mutants = sum(random.random() < self.params[1] for child in range(2))
                clones = 2 - mutants
                self.m += mutants
            else:
                mutants = 0
                clones = 0
                self.c += 1
            cumsum_clones += clones
            len_tree += 1
        assert cumsum_clones == len_tree - 1

    f_hash = {} # <--- class variable for hashing calls to the following function
    def f(self, params):
        '''
        Probability of getting c leaves that are clones of the root and m mutant clades off
        the root line, given branching probability p and mutation probability q
        Also returns gradient wrt (p, q)
        Computed by dynamic programming
        '''
        p, q = params
        c, m = self.c, self.m
        if (p, q, c, m) not in LeavesAndClades.f_hash:
            if c==m==0 or (c==0 and m==1):
                f_result = 0
                dfdp_result = 0
                dfdq_result = 0
            elif c==1 and m==0:
                f_result = 1-p
                dfdp_result = -1
                dfdq_result = 0
            elif c==0 and m==2:
                f_result = p*q**2
                dfdp_result = q**2
                dfdq_result = 2*p*q
            else:
                if m >= 1:
                    neighbor = LeavesAndClades(params=params, c=c, m=m-1)
                    neighbor_f, (neighbor_dfdp, neighbor_dfdq) = neighbor.f(params)
                    f_result = 2*p*q*(1-q)*neighbor_f
                    dfdp_result =   2*q*(1-q) * neighbor_f + \
                                  2*p*q*(1-q) * neighbor_dfdp
                    dfdq_result = (2*p - 4*p*q) * neighbor_f + \
                                   2*p*q*(1-q)  * neighbor_dfdq
                else:
                    f_result = 0.
                    dfdp_result = 0.
                    dfdq_result = 0.
                for cx in range(c+1):
                    for mx in range(m+1):
                        if (not (cx==0 and mx==0)) and (not (cx==c and mx==m)):
                            neighbor1 = LeavesAndClades(params=params, c=cx, m=mx)
                            neighbor2 = LeavesAndClades(params=params, c=c-cx, m=m-mx)
                            neighbor1_f, (neighbor1_dfdp, neighbor1_dfdq) = neighbor1.f(params)
                            neighbor2_f, (neighbor2_dfdp, neighbor2_dfdq) = neighbor2.f(params)
                            f_result += p*(1-q)**2*neighbor1_f*neighbor2_f
                            dfdp_result +=   (1-q)**2 * neighbor1_f    * neighbor2_f + \
                                           p*(1-q)**2 * neighbor1_dfdp * neighbor2_f + \
                                           p*(1-q)**2 * neighbor1_f    * neighbor2_dfdp
                            dfdq_result += -2*p*(1-q) * neighbor1_f    * neighbor2_f + \
                                           p*(1-q)**2 * neighbor1_dfdq * neighbor2_f + \
                                           p*(1-q)**2 * neighbor1_f    * neighbor2_dfdq
            LeavesAndClades.f_hash[(p, q, c, m)] = (f_result, scipy.array([dfdp_result, dfdq_result]))
        return LeavesAndClades.f_hash[(p, q, c, m)]


class CollapsedTree(LeavesAndClades):
    '''
    Here's a derived class for a collapsed tree, where we recurse into the mutant clades
          (4)
         / | \\
       (3)(1)(2)
           |   \\
          (2)  (1)
    '''
    def __init__(self, params=None, tree=None):
        '''
        For intialization, either params or tree (or both) must be provided
        params: offspring distribution parameters
        tree: ete tree with frequency node feature. If uncollapsed, it will be collapsed
        '''
        #if params is None and tree is None:
        #    raise ValueError('either params or tree (or both) must be provided')
        LeavesAndClades.__init__(self, params=params)
        if tree is not None:
            self.tree = tree.copy()
            if 0 in (node.dist for node in tree.iter_descendants()):
                # iterate over the tree below root and collapse edges of zero length
                for node in self.tree.get_descendants():
                    if node.dist == 0:
                        node.up.frequency += node.frequency
                        node.delete(prevent_nondicotomic=False)
            assert sum(node.frequency for node in tree.traverse()) == sum(node.frequency for node in self.tree.traverse())
            if 'sequence' in tree.features and len(set([node.sequence for node in self.tree.traverse()])) != sum(1 for _ in self.tree.traverse()):
                warnings.warn('repeated sequences in collapsed tree, possible backmutation', RuntimeWarning)
        else:
            self.tree = tree


    def l(self, params, sign=1):
        '''
        log likelihood of params, conditioned on collapsed tree, and its gradient wrt params
        optional parameter sign must be 1 or -1, with the latter useful for MLE by minimization
        '''
        if self.tree is None:
            raise ValueError('tree data must be defined to compute likelihood')
        if sign not in (-1, 1):
            raise ValueError('sign must be 1 or -1')
        leaves_and_clades_list = [LeavesAndClades(c=node.frequency, m=len(node.children)) for node in self.tree.traverse()]
        if leaves_and_clades_list[0].c == 0 and leaves_and_clades_list[0].m == 1 and leaves_and_clades_list[0].f(params)[0] == 0:
            # if unifurcation not possible under current model, omit root
            leaves_and_clades_list = leaves_and_clades_list[1:]
        # extract vector of function values and gradient components
        f_data = [leaves_and_clades.f(params) for leaves_and_clades in leaves_and_clades_list]
        fs = scipy.array([[x[0]] for x in f_data])
        logf = scipy.log(fs).sum()
        grad_fs = scipy.array([x[1] for x in f_data])
        grad_logf = (grad_fs/fs).sum(axis=0)
        return sign*logf, sign*grad_logf

    def mle(self, **kwargs):
        '''
        Maximum likelihood estimate for params given tree
        updates params if not None
        returns optimization result
        '''
        # random initalization
        x_0 = (random.random(), random.random())
        #x_0 = (.5, .5)
        bounds = ((.01, .99), (.001, .999))
        kwargs['sign'] = -1
        grad_check = check_grad(lambda x: self.l(x, **kwargs)[0], lambda x: self.l(x, **kwargs)[1], (.4, .5))
        if grad_check > 1e-3:
            warnings.warn('gradient mismatches finite difference approximation by {}'.format(grad_check), RuntimeWarning)
        result = minimize(lambda x: self.l(x, **kwargs), x0=x_0, jac=True, method='L-BFGS-B', options={'ftol':1e-10}, bounds=bounds)
        # update params if None and optimization successful
        if not result.success:
            warnings.warn('optimization not sucessful, '+result.message, RuntimeWarning)
        elif self.params is None:
            self.params = result.x.tolist()
        return result

    def simulate(self):
        '''
        simulate a collapsed tree given params
        replaces existing tree data member with simulation result, and returns self
        '''
        if self.params is None:
            raise ValueError('params must be defined for simulation')

        # initiate by running a LeavesAndClades simulation to get the number of clones and mutants
        # in the root node of the collapsed tree
        LeavesAndClades.simulate(self)
        self.tree = TreeNode()
        self.tree.add_feature('frequency', self.c)
        if self.m == 0:
            return self
        for _ in range(self.m):
            # ooooh, recursion
            child = CollapsedTree(params=self.params).simulate().tree
            child.dist = 1
            self.tree.add_child(child)

        return self


    def __str__(self):
        '''return a string representation for printing'''
        return 'params = ' + str(self.params)+ '\ntree:\n' + str(self.tree)

    def render(self, outfile):
        '''render to image file, filetype inferred from suffix, svg for color images'''
        for node in self.tree.traverse():
            nstyle = NodeStyle()
            if node.frequency == 0:
                nstyle['size'] = 5
                nstyle['fgcolor'] = 'grey'
            else:
                nstyle['size'] = 3*2*scipy.sqrt(scipy.pi*node.frequency)
                nstyle['fgcolor'] = 'black'
            if node.up is not None:
                if set(node.sequence.upper()) == set('ACGT'):
                    nonsyn = hamming_distance(Seq(node.sequence, generic_dna).translate(), Seq(node.up.sequence, generic_dna).translate())
                    if nonsyn > 0:
                        nstyle['hz_line_color'] = 'black'
                        nstyle['hz_line_width'] = nonsyn
                    else:
                        nstyle['hz_line_type'] = 1
                    if '*' in Seq(node.sequence, generic_dna).translate():
                        nstyle['bgcolor'] = 'red'

            node.set_style(nstyle)

        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.rotation = 90
        def my_layout(node):
            if node.frequency > 1:
                N = TextFace(node.frequency, fsize=14, fgcolor='black')
                N.rotation = -90
                faces.add_face_to_node(N, node, 0, position='branch-top')
        ts.layout_fn = my_layout
        self.tree.render(outfile, tree_style=ts)

    def write(self, file_name):
        '''serialize tree to file'''
        with open(file_name, 'wb') as f:
            cPickle.dump(self, f)


class CollapsedForest(CollapsedTree):
    '''
    simply a set of CollapsedTrees, with the same p and q parameters
          (4)          (3)
         / | \\         / \\
       (3)(1)(2)     (1) (2)
           |   \\  ,          , ...
          (2)  (1)
    '''
    def __init__(self, params=None, n_trees=None, forest=None):
        '''
        in addition to p and q, we need number of trees
        can also intialize with forest, a list of trees, each an instance of CollapsedTree
        '''
        CollapsedTree.__init__(self, params=params)
        if forest is None and params is None:
            raise ValueError('either params or forest (or both) must be provided')
        if forest is not None:
            if len(forest) == 0:
                raise ValueError('passed empty tree list')
            if n_trees is not None and len(forest) != n_trees:
                raise ValueError('n_trees not consistent with forest')
            self.forest = forest
        if n_trees is not None:
            if type(n_trees) is not int or n_trees < 1:
                raise ValueError('number of trees must be at least one')
            self.n_trees = n_trees
        if n_trees is None and forest is not None:
            self.n_trees = len(forest)

    def simulate(self):
        '''
        simulate a forest of collapsed trees given params and number of trees
        replaces existing forest data member with simulation result, and returns self
        '''
        if self.params is None or self.n_trees is None:
            raise ValueError('params and n_trees parameters must be defined for simulation')
        self.forest = [CollapsedTree(self.params).simulate() for x in range(self.n_trees)]
        return self

    def l(self, params, sign=1, Vlad_sum=False):
        '''
        likelihood of params, given forest, and it's gradient wrt params
        optional parameter sign must be 1 or -1, with the latter useful for MLE by minimization
        if optional parameter Vlad_sum is true, we're doing the Vlad sum for estimating params for
        as set of parsimony trees
        '''
        if self.forest is None:
            raise ValueError('forest data must be defined to compute likelihood')
        if sign not in (-1, 1):
            raise ValueError('sign must be 1 or -1')
        # since the l method on the CollapsedTree class returns l and grad_l...
        terms = [tree.l(params) for tree in self.forest]
        ls = scipy.array([term[0] for term in terms])
        grad_ls = scipy.array([term[1] for term in terms])
        if Vlad_sum:
            # we need to find the smallest derivative component for each
            # coordinate, then subtract off to get positive things to logsumexp
            grad_l = []
            for j in range(len(params)):
                i_prime = grad_ls[:,j].argmin()
                grad_l.append(grad_ls[i_prime,j] +
                              scipy.exp(logsumexp(ls - ls[i_prime],
                                                  b=grad_ls[:,j]-grad_ls[i_prime,j]) -
                                        logsumexp(ls - ls[i_prime])))
            return sign*(-scipy.log(len(ls)) + logsumexp(ls)), sign*scipy.array(grad_l)
        else:
            return sign*ls.sum(), sign*grad_ls.sum(axis=0)


    # NOTE: we get mle() method for free by inheritance/polymorphism magic


    def __str__(self):
        '''return a string representation for printing'''
        return 'params = {}, n_trees = {}\n'.format(self.params, self.n_trees) + \
                '\n'.join([str(tree) for tree in self.forest])


def hamming_distance(seq1, seq2):
    '''Hamming distance between two sequences of equal length'''
    return sum(x != y for x, y in zip(seq1, seq2))


def phylip_parse(phylip_outfile, naive=None):
    '''parse phylip outfile and return ete trees'''
    # parse phylip outfile
    outfiledat = [block.split('\n\n\n')[0].split('\n\n') for block in open(phylip_outfile, 'r').read().split('From    To     Any Steps?    State at upper node')[1:]]

    # ete trees
    trees = []
    for i, tree in enumerate(outfiledat):
        tree_sequence_dict = {}
        parent_dict = {}
        names = []
        for j, block in enumerate(tree):
            if j == 0:
                for line in block.split('\n'):
                    fields = line.split()
                    if len(fields) == 0:
                        continue
                    name = fields[1]
                    names.append(name)
                    if fields[0] == 'root':
                        seq = ''.join(fields[2:])
                        parent = None
                    else:
                        seq = ''.join(fields[3:])
                        parent = fields[0]
                    tree_sequence_dict[name] = seq
                    parent_dict[name] = parent
            else:
                for line in block.split('\n'):
                    fields = line.split()
                    name = fields[1]
                    if fields[0] == 'root':
                        seq = ''.join(fields[2:])
                    else:
                        seq = ''.join(fields[3:])
                    tree_sequence_dict[name] += seq

        # if integer branch (not weird ambiguous chars)
        if set(''.join([tree_sequence_dict[name] for name in names])) == set('ACGT'):
            #nodes = dict([(name, Tree(name=(name, tree_sequence_dict[name]), dist=hamming_distance(tree_sequence_dict[name], tree_sequence_dict[parent_dict[name]]) if parent_dict[name] is not None else None)) for name in names])
            nodes = {}
            for name in names:
                node = TreeNode()
                node.name = name
                node.dist = hamming_distance(tree_sequence_dict[name], tree_sequence_dict[parent_dict[name]]) if parent_dict[name] is not None else 0
                node.add_feature('sequence', tree_sequence_dict[node.name])
                if node.name == naive:
                    node.add_feature('frequency', 0)
                elif '_' in node.name:
                    node.add_feature('frequency', int(node.name.split('_')[-1]))
                    node.name = '_'.join(node.name.split('_')[:-1])
                else:
                    node.add_feature('frequency', 0)
                nodes[name] = node
            tree = nodes[names[0]] # naive is first
            for name in parent_dict:
                if parent_dict[name] is not None:
                    nodes[parent_dict[name]].add_child(nodes[name])
            # reroot on naive
            if naive is not None:
                assert len(nodes[naive].children) == 0
                assert nodes[naive] in tree.children
                tree.remove_child(nodes[naive])
                nodes[naive].add_child(tree)
                tree.dist = nodes[naive].dist
                tree = nodes[naive]
                tree.dist = 0

            # assert branch lengths make sense
            for node in tree.iter_descendants():
                assert node.dist == hamming_distance(node.sequence, node.up.sequence)

            trees.append(tree)

    return trees


class MutationModel():
    '''a class for a mutation model, and functions to mutate sequences'''
    def __init__(self, mutability_file, substitution_file):
        '''initialized with input files of the S5F format'''
        self.mutation_model = {}
        with open(mutability_file, 'r') as f:
            # eat header
            f.readline()
            for line in f:
                motif, score = line.replace('"', '').split()[:2]
                self.mutation_model[motif] = float(score)

        # kmer k
        self.k = None
        with open(substitution_file, 'r') as f:
            # eat header
            f.readline()
            for line in f:
                fields = line.replace('"', '').split()
                motif = fields[0]
                if self.k is None:
                    self.k = len(motif)
                    assert self.k % 2 == 1
                else:
                    assert len(motif) == self.k
                self.mutation_model[motif] = (self.mutation_model[motif], {b:float(x) for b, x in zip('ACGT', fields[1:5])})

    def mutability(self, kmer):
        '''returns the mutability of a kmer, along with nucleotide biases'''
        assert len(kmer) == self.k
        return self.mutation_model[kmer]

    def mutate(self, sequence, lambda0=1):
        '''mutate a sequence, with q the baseline mutability'''
        assert all(n in 'ACGT' for n in sequence)
        # mutabilities of each nucleotide
        mutabilities = []
        sequence_length = len(sequence)
        assert sequence_length >= 5
        # ambiguous left end motifs
        for i in range(self.k//2 + 1):
            kmer_suffix = sequence[:(i+self.k//2+1)]
            matches = [value for key, value in self.mutation_model.iteritems() if key.endswith(kmer_suffix)]
            len_matches = len(matches)
            assert len_matches == 4**(self.k - len(kmer_suffix))
            # use mean over matches
            mutability = sum(match[0] for match in matches)/len_matches
            substitution = {n:sum(d[1][n] for d in matches)/len_matches for n in 'ACGT'}
            mutabilities.append((mutability, substitution))
        # unambiguous internal kmers
        for i in range(self.k//2, sequence_length - self.k//2):
            mutabilities.append(self.mutability(sequence[(i-self.k//2):(i+self.k//2+1)]))
        # ambiguous right end motifs
        for i in range(sequence_length - self.k//2 + 1, sequence_length):
            kmer_prefix = sequence[(i-self.k//2):]
            matches = [value for key, value in self.mutation_model.iteritems() if key.startswith(kmer_prefix)]
            len_matches = len(matches)
            assert len_matches == 4**(self.k - len(kmer_prefix))
            # use mean over matches
            mutability = sum(match[0] for match in matches)/float(len_matches)
            substitution = {n:sum(d[1][n] for d in matches)/float(len_matches) for n in 'ACGT'}
            mutabilities.append((mutability, substitution))

        assert len(mutabilities) == sequence_length

        # mean mutability
        sequence_mutability = sum(mutability[0] for mutability in mutabilities)/float(sequence_length)
        # baseline Piosson
        #lambda_0 = -scipy.log(1-q)
        # poisson rate for this sequence (given its relative mutability)
        lambda_sequence = sequence_mutability*lambda0
        # number of mutations
        m = scipy.random.poisson(lambda_sequence)

        if m > 0:
            # now we choose random positions for the m mutations, weighted by mutabilities
            # invoking a long sequence limit, we don't allow back mutations
            # draw a multinomial rv for the number of mutations in each site
            p = [mutability[0]/(sequence_length*sequence_mutability) for mutability in mutabilities]
            assert 0 <= abs(sum(p) - 1.) < 1e-10
            mutated_sites = scipy.random.multinomial(m, p)
            trial = 0
            while max(mutated_sites) > 1:
                print('repeated mutations, trying again')
                trial += 1
                if trial > 5:
                    raise RuntimeError('mutations saturating')
                mutated_sites = scipy.random.multinomial(m, p)
            sequence = list(sequence) # mutable
            for i in range(sequence_length):
                if mutated_sites[i]:
                    p = [mutabilities[i][1][n] for n in 'ACGT']
                    assert 0 <= abs(sum(p) - 1.) < 1e-10
                    sequence[i] = 'ACGT'[scipy.nonzero(scipy.random.multinomial(1, p))[0][0]]
            sequence = ''.join(sequence)

        return sequence


    def simulate(self, sequence, p=.4, lambda0=1, r=1.):
        '''simulate neutral binary branching process with mutation model'''
        if p >= .5:
            raw_input('WARNING: p = {} is not subcritical, tree termination not garanteed! [ENTER] to proceed'.format(p))
        tree = TreeNode()
        tree.dist = 0
        tree.add_feature('sequence', sequence)
        tree.add_feature('terminated', False)
        tree.add_feature('frequency', 0)

        nodes_unterminated = 1
        while nodes_unterminated > 0:
            for leaf in tree.iter_leaves():
                if not leaf.terminated:
                    if scipy.random.random() < p:
                        for child_count in range(2):
                            mutated_sequence = self.mutate(leaf.sequence, lambda0=lambda0)
                            child = TreeNode()
                            child.dist = sum(x!=y for x,y in zip(mutated_sequence, leaf.sequence))
                            child.add_feature('sequence', mutated_sequence)
                            child.add_feature('frequency', 0)
                            leaf.add_child(child)
                            child.add_feature('terminated' ,False)
                        nodes_unterminated += 1
                    else:
                        leaf.terminated = True
                        nodes_unterminated -= 1

        # each leaf gets an observation frequency of 1
        for node in tree.iter_leaves():
            if scipy.random.random() < r:
                node.frequency = 1

        # return the fine (uncollapsed) tree
        return tree


def test(args):
    '''
    test subprogram
    checks likelihood against a by-hand calculation for a simple tree, simulates a forest, computes MLE parameters, and plots some sanity check figures to plot_file
    command line arguments are p, q, number of trees to simulate, and plot file name
    '''
    p = args.p
    q = args.q
    n = args.n
    plot_file = args.outbase

    if plot_file[-4:] != '.pdf':
        plot_file += '.pdf'

    print('Let''s check our likelihood against a by-hand calculation for the following simple tree')
    # ete tree
    parent = TreeNode(format=1)
    parent.add_feature('frequency', 2)
    parent.name = parent.frequency
    child = TreeNode()
    child.add_feature('frequency', 1)
    child.dist = 1
    child.name = child.frequency
    parent.add_child(child)
    tree = CollapsedTree(tree=parent)
    f = 6*p**2*(1-p)**3*q*(1-q)**3
    dfdp = 6*(1 - p)**2*p*(-2 + 5*p)*(-1 + q)**3*q #6*q*(1-q)**3*(2*p*(1-p)**3-3*p**2*(1-p)**2)
    dfdq = 6*(-1 + p)**3*p**2*(1 - q)**2*(-1 + 4*q) #6*p**2*(1-p)**3*((1-q)**3-3*q*(1-q)**2)
    print( '    T =', tree.tree.get_ascii(show_internal=True))
    print( '    Summing the probabilities of the two possible fine structures, we have')
    print( '    logP =', scipy.log(f))
    print(u'    \u2207logP = ', (dfdp/f, dfdq/f))
    print( '    Now, our dynamic programming algorithm gives')
    print(u'    logP , \u2207logP =', tree.l((p, q)))
    print('')

    print('Simulating a forest of {} trees'.format(n))
    forest = CollapsedForest((p, q), n)
    print('    true parameters: p = {}, q = {}'.format(p, q))
    forest.simulate()

    # total leaf counts
    total_data = sorted([sum(node.frequency for node in tree.tree.traverse()) for tree in forest.forest])
    max_total = max(total_data)
    len_total = len(total_data)

    totals = []
    freq = []
    log_prob = []
    for x in range(1, max_total+1):
        totals.append(x)
        freq.append(total_data.count(x))
        tmp_tree = TreeNode(format=1)
        tmp_tree.add_feature('frequency', x)
        log_prob.append(CollapsedTree(tree=tmp_tree).l((p, 0))[0])
    theoretical_cdf = scipy.cumsum(scipy.exp(log_prob))
    empirical_cdf = scipy.cumsum(freq)/float(len_total)

    fig = plt.figure()
    fig.set_tight_layout(True)
    plt.rc('text', usetex=True)

    # plot the empirical and theoretical distribution of total leaf counts

    ax = fig.add_subplot(2,2,1)
    ax.plot(totals, scipy.exp(log_prob), 'ko', markerfacecolor='none', alpha=.5, label='theoretical PMF')
    ax.plot(totals, scipy.array(freq)/float(len_total), 'k.', label='empirical PMF')
    ax.legend(numpoints=1, loc=1, fontsize='small')
    ax.set_xlabel('total leaves')
    ax.set_ylabel('$\Pr($total leaves$)$')
    ax.set_ylim([0, 1.1])
    #ax.set_xscale('log')
    #ax.set_yscale('symlog')

# uncomment this if you want the CDF
#    ax = fig.add_subplot(2,2,2)
#    ax.plot(totals, theoretical_cdf, 'ko', markerfacecolor='none', alpha=.5, label='theoretical CDF')
#    ax.plot(totals, empirical_cdf, 'k.', label='empirical CDF')
#    ax.legend(numpoints=1, loc=4, fontsize='small')
#    ax.set_xlabel('number of leaves')
#    ax.set_ylim([0, 1.1])


    empirical_quantiles = []
    theoretical_quantiles = []
    for x in total_data:
        empirical_quantiles.append(sum(y <= x for y in total_data)/float(len_total))
        to_add = 0.
        for y in range(1, x+1):
            tmp_tree = TreeNode(format=1)
            tmp_tree.add_feature('frequency', y)
            to_add += scipy.exp(CollapsedTree(tree=tmp_tree).l((p, 0))[0])
        theoretical_quantiles.append(to_add)

    ax = fig.add_subplot(2,2,2)
    ax.plot(theoretical_quantiles, empirical_quantiles, 'ko', alpha=.1)
    ax.plot([0, 1], [0, 1], 'k')
    ax.set_title('total leaves')
    ax.set_xlabel('theoretical quantiles')
    ax.set_ylabel('empirical quantiles')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')

    mle = forest.mle()

    #for tree in forest.forest:
    #    print(tree)
    print('    MLE parameters:  p = {}, q = {}'.format(*mle.x.tolist()))

    # plot the 2-norm of the difference between the gradient and its finite difference approximation
    print('computing plot data...')
    X, Y = scipy.mgrid[slice(.05, 1, .05),
                       slice(.05, 1, .05)]
    Z = scipy.zeros((X.shape[0], X.shape[1]))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = check_grad(lambda x: forest.l(x)[0], lambda x: forest.l(x)[1], (X[i, j], Y[i, j]))

    print('done')
    ax = fig.add_subplot(2,2,3)
    ax.set_title(r'$||\nabla \ell(p, q) - \Delta \ell(p, q)||_2$')
    im = ax.contourf(X, Y, Z, locator=ticker.LogLocator(), cmap='Greys')
    ax.set_xlabel(r'$p$')
    ax.set_ylabel(r'$q$')
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax)


    # plot likelihood surface, with true and MLE parameters shown
    X, Y = scipy.mgrid[slice(.02, 1, .02),
                       slice(.02, 1, .02)]
    Z = scipy.zeros((X.shape[0], X.shape[1]))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            l, grad_l = forest.l((X[i, j], Y[i, j]))
            z = l
            Z[i, j] = z
    ax = fig.add_subplot(2,2,4)
    ax.set_title(r'$\ell(p, q)$')
    contour = ax.contour(X, Y, Z, colors='k', label='likelihood contours')
    for c in contour.collections:
        c.set_linestyle('solid')

    ax.clabel(contour, fontsize=9, inline=1)
    ax.plot([p], [q], 'k+', label='true parameters')
    ax.plot(mle.x[0], mle.x[1], 'ko', markerfacecolor='none', label='MLE parameters')
    ax.set_xlabel(r'$p$')
    ax.set_ylabel(r'$q$')
    ax.set_aspect('equal')
    ax.legend(numpoints = 1, fontsize='small')

    plt.savefig(plot_file)
    print('plot saved to', plot_file)


def infer(args):
    '''inference subprogram'''
    forest = CollapsedForest(forest=[CollapsedTree(tree=tree) for tree in phylip_parse(args.phylipfile, args.naive)])

    if forest.n_trees == 1:
        warnings.warn('only one parsimony tree reported from dnapars')

    print('number of trees with integer branch lengths:', forest.n_trees)

    # check for unifurcations at root
    unifurcations = sum(tree.tree.frequency == 0 and len(tree.tree.children) == 1 for tree in forest.forest)
    if unifurcations:
        print('WARNING: {} trees exhibit unifurcation from root, which is not possible under current model. Such nodes will be ommitted from likelihood calculation'.format(unifurcations))

    # fit p and q using all trees
    # if we get floating point errors, try a few more times (starting params are random)
    max_tries = 1
    for tries in range(max_tries):
        try:
            forest.mle(Vlad_sum=True)
            break
        except FloatingPointError as e:
            if tries + 1 < max_tries:
                print('floating point error in MLE: {}. Attempt {} of {}. Rerunning with new random start.'.format(e, tries+1, max_tries))
            else:
                raise
        else:
            raise

    print('params = {}'.format(forest.params))

    with open(args.outbase+'.inference.parsimony_forest.p', 'wb') as f:
        cPickle.dump(forest, f)

    print_data = []
    max_l = None
    for collapsed_tree in forest.forest:
        alleles = len(collapsed_tree.tree)
        l = collapsed_tree.l(forest.params)[0]
        if max_l is None or l > max_l:
            mle_tree = collapsed_tree
        print_data.append((alleles, l))
    mle_tree.render(args.outbase+'.inference.MLtree.svg')

    print('alleles\tlogLikelihood')
    for x in sorted(print_data, key=lambda x: (-x[-1], x[0])):
        print('\t'.join(map(str, x)))


def simulate(args):
    '''simulation subprogram'''
    if args.lambda0 is None:
        args.lambda0 = max([1, int(.01*len(args.sequence))])
    args.sequence = args.sequence.upper()
    mutation_model = MutationModel(args.mutability, args.substitution)
    size = 0
    while size < args.n:
        tree = mutation_model.simulate(args.sequence, p=args.p, lambda0=args.lambda0, r=args.r)
        size = sum(node.frequency for node in tree)
    with open(args.outbase+'.simulation.fasta', 'w') as f:
        f.write('> naive\n')
        f.write(args.sequence+'\n')
        i = 0
        for leaf in tree.iter_leaves():
            if leaf.frequency != 0:# and '*' not in Seq(leaf.sequence, generic_dna).translate():
                i += 1
                f.write('> seq{}\n'.format(i))
                f.write(leaf.sequence+'\n')
                leaf.name = 'seq{}'.format(i)
    print(i, 'simulated observed sequences')
    #tree.render(args.outbase+'.tree.svg')
    collapsed_tree = CollapsedTree(tree=tree)
    collapsed_tree.write( args.outbase+'.simulation.collapsed_tree.p')
    collapsed_tree.render(args.outbase+'.simulation.collapsed_tree.svg')


def validate(args):
    with open(args.truetree, 'rb') as f:
        true_tree = cPickle.load(f)
    with open(args.parfor, 'rb') as f:
        parsimony_forest = cPickle.load(f)

    distances, likelihoods = zip(*[(true_tree.tree.robinson_foulds(tree.tree, attr_t1='sequence', attr_t2='sequence')[0],
                                    tree.l(parsimony_forest.params)[0]) for tree in parsimony_forest.forest])

    df = pd.DataFrame({'distance':distances, 'log-likelihood':likelihoods})
    df.to_csv(args.outbase+'.validation.tsv', sep='\t', index=False)
    maxl_idx = df['log-likelihood'].idxmax()
    minl_idx = df['log-likelihood'].idxmin()
    d_ranks = df['distance'].rank(method='min')
    print('parsimony forest size: {}'.format(len(df.index)))
    print('                l_max: {}'.format(df['log-likelihood'][maxl_idx]))
    print('      d of l_max tree: {}'.format(df['distance'][maxl_idx]))
    print(' rank d of l_max tree: {}'.format(d_ranks[maxl_idx]))
    print('                l_min: {}'.format(df['log-likelihood'][minl_idx]))
    print('      d of l_min tree: {}'.format(df['distance'][minl_idx]))
    print(' rank d of l_min tree: {}'.format(d_ranks[minl_idx]))
    sns.regplot(x='log-likelihood', y='distance', data=df).get_figure().savefig(args.outbase+'.validation.pdf')

def main():
    import argparse

    parser = argparse.ArgumentParser(description='germinal center tree inference and simulation')
    subparsers = parser.add_subparsers(help='which program to run')

    # parser for test subprogram
    parser_test = subparsers.add_parser('test', help='run tests on library functions')
    parser_test.add_argument('--p', type=float, default=.4, help='branching probability for test mode')
    parser_test.add_argument('--q', type=float, default=.5, help='mutation probability for test mode')
    parser_test.add_argument('--n', type=int, default=100, help='forest size for test mode')
    parser_test.set_defaults(func=test)

    # parser for inference subprogram
    parser_infer = subparsers.add_parser('infer', help='likelihood ranking of parsimony trees')
    parser_infer.add_argument('--naive', type=str, default=None, help='name of naive sequence (outgroup root)')
    parser_infer.add_argument('phylipfile', type=str, help='dnapars outfile (verbose output with sequences at each site)')
    parser_infer.set_defaults(func=infer)

    # parser for simulation subprogram
    parser_sim = subparsers.add_parser('simulate', help='neutral model gctree simulation')
    parser_sim.add_argument('sequence', type=str, help='seed naive nucleotide sequence')
    parser_sim.add_argument('mutability', type=str, help='path to mutability model file')
    parser_sim.add_argument('substitution', type=str, help='path to substitution model file')
    parser_sim.add_argument('--p', type=float, default=.4, help='branching probability')
    parser_sim.add_argument('--lambda0', type=float, default=None, help='baseline mutation rate')
    parser_sim.add_argument('--r', type=float, default=1., help='sampling probability')
    parser_sim.add_argument('--n', type=int, default=1, help='minimum simulation size')
    parser_sim.set_defaults(func=simulate)

    # parser for validation subprogram
    parser_val = subparsers.add_parser('validate', help='validate results of inference on simulation data')
    parser_val.add_argument('truetree', type=str, help='.p file containing true tree')
    parser_val.add_argument('parfor', type=str, help='.p file containing parsimony forest from inference')
    parser_val.set_defaults(func=validate)

    # a common outbase parameter
    for subparser in [parser_test, parser_infer, parser_sim, parser_val]:
        subparser.add_argument('--outbase', type=str, default='gctree.out', help='output file base name')


    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
