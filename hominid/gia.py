import itertools
import numpy as np
import matplotlib.pyplot as plt


class GlobalImportance():
    """Class that performs GIA experiments."""
    def __init__(self, model, alphabet='ACGU'):
        self.model = model
        self.alphabet = alphabet
        self.x_null = None
        self.x_null_index = None


    def set_null_model(self, null_model, base_sequence, num_sample=1000, base_scores=None):
        """use model-based approach to set the null sequences"""
        self.x_null = generate_null_sequence_set(null_model, base_sequence, num_sample, base_scores) 
        self.x_null_index = np.argmax(self.x_null, axis=2)
        self.predict_null()


    def set_x_null(self, x_null):
        """set the null sequences"""
        self.x_null = x_null
        self.x_null_index = np.argmax(x_null, axis=2)
        self.predict_null()


    def filter_null(self, low=10, high=90, num_sample=1000):
        """ remove sequences that yield extremum predictions"""
        high = np.percentile(self.null_scores, high)
        low = np.percentile(self.null_scores, low)
        index = np.where((self.null_scores < high)&(self.null_scores > low))[0]
        self.set_x_null(self.x_null[index][:num_sample])
        self.predict_null()

               
    def predict_null(self, class_index=0):
        """perform GIA on null sequences"""
        self.null_scores = self.model.predict(self.x_null)[:, class_index]
        self.mean_null_score = np.mean(self.null_scores)


    def embed_patterns(self, patterns):
        """embed patterns in null sequences"""
        if not isinstance(patterns, list):
            patterns = [patterns]

        x_index = np.copy(self.x_null_index)
        for pattern, position in patterns:

            # convert pattern to categorical representation
            pattern_index = np.array([self.alphabet.index(i) for i in pattern])

            # embed pattern 
            x_index[:,position:position+len(pattern)] = pattern_index

        # convert to categorical representation to one-hot 
        one_hot = np.zeros((len(x_index), len(x_index[0]), len(self.alphabet)))
        for n, x in enumerate(x_index):
            for l, a in enumerate(x):
                one_hot[n,l,a] = 1.0

        return one_hot
    

    def embed_predict_effect(self, patterns, class_index=0, subtract_null=True):
        """embed pattern in null sequences and get their predictions"""
        one_hot = self.embed_patterns(patterns)
        predictions = self.model.predict(one_hot)[:, class_index] 
        if subtract_null:
            return predictions - self.null_scores
        else:
            return predictions

    def predict_effect(self, one_hot, class_index=0, subtract_null=True):
        """Measure effect size of sequences versus null sequences"""
        predictions = self.model.predict(one_hot)[:, class_index]
        if subtract_null:
            return predictions - self.null_scores
        else:
            return predictions

        
    def optimal_flanks(self, pattern, position, class_index=0, subtract_null=True):
        """GIA to find optimal flanks"""

        # generate all kmers  
        index = []
        for i, a in enumerate(pattern):
            if a == 'N':
                index.append(i)
        flank_size = len(index)
        kmers = ["".join(p) for p in itertools.product(list(self.alphabet), repeat=flank_size)]

        # score each kmer
        mean_scores = []
        all_patterns = []
        for kmer in kmers:
            embed_pattern = list(pattern)
            for i in range(flank_size):
                embed_pattern[index[i]] = kmer[i]
            embed_pattern = "".join(embed_pattern)
            effect = self.embed_predict_effect((embed_pattern, position), class_index, subtract_null)
            mean_scores.append(np.mean(effect))
            all_patterns.append(embed_pattern)
        mean_scores = np.array(mean_scores)
        all_patterns = np.array(all_patterns)

        # sort by highest prediction
        sort_index = np.argsort(mean_scores)[::-1]

        return mean_scores[sort_index], all_patterns[sort_index]



    def optimal_position(self, motif, start, end, class_index=0, subtract_null=True):
        """GIA to find positional bias"""

        # loop over positions and measure effect size of intervention
        positions = range(start, end)
        all_scores = []
        for position in positions:
            score = self.embed_predict_effect([(motif, position)], class_index, subtract_null)
            all_scores.append(score)
        return np.array(all_scores), np.array(positions)
    

    def positional_dependence(self, motif1, position1, motif2, start, end, class_index=0, subtract_null=True):
        """GIA to find positional bias"""

        # loop over positions and measure effect size of intervention
        positions = range(start, end)
        all_scores = []
        for position2 in positions:
            interventions = [(motif1, position1), (motif2, position2)]
            score = self.embed_predict_effect(interventions, class_index, subtract_null)
            all_scores.append(score)
        return np.array(all_scores), np.array(positions)

    
    def interactions(self, motif1, position1, motif2, position2, class_index=0, subtract_null=True):
        motif1_pred = self.embed_predict_effect((motif1, position1), class_index, subtract_null)
        motif2_pred = self.embed_predict_effect((motif2, position2), class_index, subtract_null)
        interventions = [(motif1, position1), (motif2, position2)]
        both_pred = self.embed_predict_effect(interventions, class_index, subtract_null)
        return both_pred, motif1_pred, motif2_pred
   


#-------------------------------------------------------------------------------------
# Null sequence models
#-------------------------------------------------------------------------------------

    
def generate_null_sequence_set (null_model, base_sequence, num_sample=1000 , base_scores=None):
    if null_model == 'random':    return generate_shuffled_set(base_sequence, num_sample)
    if null_model == 'profile':   return generate_profile_set(base_sequence, num_sample)
    if null_model == 'dinuc':     return generate_dinucleotide_shuffled_set(base_sequence, num_sample)
    if null_model == 'quartile1': return generate_quartile_set(base_sequence, num_sample, base_scores, quartile=1)
    if null_model == 'quartile2': return generate_quartile_set(base_sequence, num_sample, base_scores, quartile=2)
    if null_model == 'quartile3': return generate_quartile_set(base_sequence, num_sample, base_scores, quartile=3)
    if null_model == 'quartile4': return generate_quartile_set(base_sequence, num_sample, base_scores, quartile=4)
    else: print ('null_model name not recognized.')


def generate_profile_set(base_sequence, num_sample):
    # set null sequence model
    seq_model = np.mean(np.squeeze(base_sequence), axis=0) 
    seq_model /= np.sum(seq_model, axis=1, keepdims=True) 

    # sequence length
    L = seq_model.shape[0]

    x_null = np.zeros((num_sample, L, 4))
    for n in range(num_sample):

        # generate uniform random number for each nucleotide in sequence
        Z = np.random.uniform(0,1,L)

        # calculate cumulative sum of the probabilities
        cum_prob = seq_model.cumsum(axis=1)

        # find bin that matches random number for each position
        for l in range(L):
            index = [j for j in range(4) if Z[l] < cum_prob[l,j]][0]
            x_null[n,l,index] = 1    
    return x_null 


def generate_shuffled_set(base_sequence, num_sample): 
    # take a random subset of base_sequence
    shuffle = np.random.permutation(len(base_sequence))
    x_null = base_sequence[shuffle[:num_sample]]

    # shuffle nucleotides   
    [np.random.shuffle(x) for x in x_null]
    return x_null   


def generate_dinucleotide_shuffled_set(base_sequence, num_sample):  

    # take a random subset of base_sequence
    shuffle = np.random.permutation(len(base_sequence))
    x_null = base_sequence[shuffle[:num_sample]]
    
    # shuffle dinucleotides       
    for j, seq in enumerate(x_null): 
        x_null[j] = dinuc_shuffle(seq)
    return x_null    


def generate_quartile_set(base_sequence, num_sample, base_scores, quartile): 
    # sort sequences by the binding score (descending order)
    sort_index = np.argsort(base_scores[:,0])[::-1]
    base_sequence = base_sequence[sort_index]
    
    # set quartile indices
    L = len(base_sequence)
    L0, L1, L2, L3, L4 = [0, int(L/4), int(L*2/4), int(L*3/4), L]
    
    # pick the quartile:
    if (quartile==1): base_sequence = base_sequence[L0:L1]
    if (quartile==2): base_sequence = base_sequence[L1:L2]
    if (quartile==3): base_sequence = base_sequence[L2:L3]
    if (quartile==4): base_sequence = base_sequence[L3:L4]
    
    # now shuffle the sequences
    shuffle = np.random.permutation(len(base_sequence))
   
    # take a smaller sample of size num_sample
    return base_sequence[shuffle[:num_sample]]
    

    
def optimal_interactions(gia, motif1, motif1_index, motif2, window_scan=50, class_index=0):

    # positional dependence plot
    start = motif1_index - window_scan
    end = motif1_index + window_scan
    interaction_pos_scores, interaction_pos = gia.positional_dependence(motif1, motif1_index, motif2, start, end, class_index)
    motif2_index = start + np.argmax(np.mean(interaction_pos_scores, axis=1))

    # interactions at optimal distances
    scores12, scores1, scores2 = gia.interactions(motif1, motif1_index, motif2, motif2_index, class_index)

    return (scores12, scores1, scores2), (interaction_pos_scores, interaction_pos)

