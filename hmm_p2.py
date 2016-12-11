#! /usr/bin/python

__author__="Angela Mayhua <amayhuaq@gmail.com>"
__date__ ="$Dec 05, 2016"

import sys
from collections import defaultdict
import math


def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            ne_tag = fields[-1]
            #phrase_tag = fields[-2] #Unused
            #pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else: # Empty line
            yield (None, None)                        
        l = corpus_file.readline()

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
         #Add boundary symbols to the sentence
         w_boundary = (n-1) * [(None, "*")]
         w_boundary.extend(sent)
         w_boundary.append((None, "STOP"))
         #Then extract n-grams
         ngrams = (tuple(w_boundary[i:i+n]) for i in xrange(len(w_boundary)-n+1))
         for n_gram in ngrams: #Return one n-gram at a time
            yield n_gram        


def combinations(list_a, list_b):
    for a in list_a:
        for b in list_b:
            yield (a, b)


class Hmm(object):
    """
    Stores counts for n-grams and emissions. 
    """

    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.words_count = {}
        self.emission_counts = defaultdict(int)
        self.emission_probs = defaultdict(float)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()


    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(corpus_file)), self.n)

        for ngram in ngram_iterator:
            #Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram]) #retrieve only the tags            
            for i in xrange(2, self.n+1): #Count NE-tag 2-grams..n-grams
                self.ngram_counts[i-1][tagsonly[-i:]] += 1
            
            if ngram[-1][0] is not None: # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1 # count 1-gram
                self.emission_counts[ngram[-1]] += 1 # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None: # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

    def write_counts(self, output, printngrams=[1,2,3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:            
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))

        # Then write counts for all ngrams
        for n in printngrams:            
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %(self.ngram_counts[n-1][ngram], n, ngramstr))


    def read_counts(self, corpusfile):
        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n-1][ngram] = count


    def word_count(self, corpusfile):
        """
        Counts the word in the file to delete infrequent words.
        """
        self.words_count = {}

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                word = parts[3]
                if word in self.words_count:
                    self.words_count[word] += count;
                else:
                    self.words_count[word] = count;


    def clean_data(self, trainFile, out):
        """
        Infrequent words change their text to _RARE_ when their count < 5.
        Create new file with cleaned words
        """
        for line in trainFile:
            parts = line.strip().split(" ")

            if len(parts) == 2:
                word = parts[0]
                ne_tag = parts[1]

                if self.words_count[word] < 5:
                    line = "_RARE_ " + ne_tag + "\n"
            out.write(line)


    def compute_emission_prob(self):
        """
        Compute the emission probabilities per each word of the set
        """
        self.emission_probs = defaultdict(float)
        for word, ne_tag in self.emission_counts:
            self.emission_probs[(word, ne_tag)] = self.emission_counts[(word, ne_tag)] / self.ngram_counts[0][tuple([ne_tag,])]


    def compute_trigram_prob(self):
        """
        Compute the trigram probabilities
        """
        self.trigram_probs = defaultdict(float)
        for ngram in self.ngram_counts[self.n-1]:
            ngramstr = " ".join(ngram)
            cls = ngramstr.split()
            ngram2 = tuple(cls[:2])
            self.trigram_probs[(cls[-1], ngram2)] = self.ngram_counts[-1][ngram] / self.ngram_counts[-2][ngram2]


    def get_emission_rare(self, word, tag):
        return self.emission_probs[('_RARE_',tag)]
        
        
    def S(self, k):
        """
        Define the set of available tags when we are at the k position in the sentence
        """
        if k in (-2,-1): 
            return ['*']
        return self.all_states
    
    
    def dinamic_viterbi(self, sentence, labels):
        """
        Viterbi algorithm that tag each word in the sentence computing the maximum probability at the k position
        """
        n = len(sentence)
        pi = {}
        bp = {}
        pi[(-1,'*','*')] = 1.0
        
        for k in range(0, n):
            for u, v in combinations(self.S(k-1), self.S(k)):
                pi[(k,u,v)], bp[(k,u,v)] = max([(pi[(k-1,w,u)] * self.trigram_probs[(v,(w,u))] * self.emission_probs[(sentence[k],v)], w) for w in self.S(k-2)])
                if pi[(k,u,v)] == 0:
                    pi[(k,u,v)], bp[(k,u,v)] = max([(pi[(k-1,w,u)] * self.trigram_probs[(v,(w,u))] * self.get_emission_rare(sentence[k],v), w) for w in self.S(k-2)])
                #print (k,u,v), pi[(k,u,v)], bp[(k,u,v)]
        #for u, v in combinations(self.S(n-2), self.S(n-1)):
        #    print (u,v), self.trigram_probs[('STOP',(u,v))]
            
        _, u, v = max([(pi[(n-1,u1,v1)] * self.trigram_probs[('STOP',(u1,v1))], u1, v1) for u1,v1 in combinations(self.S(n-2), self.S(n-1))])
        #print (u,v)

        if n >= 1:
            labels[n-1] = v
        if n >= 2:
            labels[n-2] = u
            
        for k in range(n-3, -1, -1):
            labels[k] = bp[k+2, labels[k+1], labels[k+2]]


    def tag_words2(self, testFile, out):
        """
        We tag the words inside testFile and write the results in an output file.
        """
        list_sentences = sentence_iterator(simple_conll_corpus_iterator(testFile))
        for sentence in list_sentences:
            words = [word for ne_tag, word in sentence]
            labels = ['' for w in words]
            self.dinamic_viterbi(words, labels)
            for j in xrange(len(labels)):
                out.write(words[j] + " " + labels[j] + "\n")
            out.write("\n")


def usage():
    print """
    python count_freqs.py [train_file] [counts_file] [test_file] [output_file]
        Read in a gene tagged training input file and produce counts.
    """

if __name__ == "__main__":

    if len(sys.argv) != 5:
        usage()
        sys.exit(2)

    try:
        trainFile = file(sys.argv[1],"r")
        countsFile = file(sys.argv[2],"w")
        testFile = file(sys.argv[3],"r")
        outFile = file(sys.argv[4],"w")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)
    
    # Initialize a trigram counter
    counter = Hmm(3)
    # Collect counts
    counter.train(trainFile)
    # Write the counts
    counter.write_counts(countsFile)
    # Counts words
    countsFile = file(sys.argv[2],"r")
    counter.word_count(countsFile)
    # Clean the infrequent words from data file
    trainFile = file(sys.argv[1],"r")
    cleaned = file("gene.train.cleaned","w")
    counter.clean_data(trainFile, cleaned)
    
    counter = Hmm(3)
    # Recollect counts
    cleaned = file("gene.train.cleaned","r")
    counter.train(cleaned)
    # Rewrite the counts
    countsFile = file(sys.argv[2],"w")
    counter.write_counts(countsFile)
    # Read the counts
    countsFile = file(sys.argv[2],"r")
    counter.read_counts(countsFile)
    # Compute probabilities
    counter.compute_emission_prob()
    counter.compute_trigram_prob()
    # Tag of words
    counter.tag_words2(testFile,outFile)
