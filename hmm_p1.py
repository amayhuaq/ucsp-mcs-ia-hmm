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
                ne_tag = parts[2]
                word = parts[3]
                if word in self.words_count:
                    self.words_count[word] += count;
                else:
                    self.words_count[word] = count;


    def clean_data(self, trainFile, out):
        """
        Infrequent words change their tag to _RARE_ when their count < 5.
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


    def tag_words1(self, testFile, out):
        """
        We tag the words inside testFile and write the results in an output file.
        """
        for word in testFile:
            word = word.strip()
            valTag = 0
            tag = ""
            if len(word) > 0:
                for ne_tag in self.all_states:
                    if self.emission_probs[(word, ne_tag)] > valTag:
                        valTag = self.emission_probs[(word, ne_tag)]
                        tag = ne_tag
                if tag == "":
                    for ne_tag in self.all_states:
                        if self.emission_probs[("_RARE_", ne_tag)] > valTag:
                            valTag = self.emission_probs[("_RARE_", ne_tag)]
                            tag = ne_tag
                    
                word = word + " " + tag
            out.write(word + "\n")

              
def usage():
    print """
    python count_freqs.py [train_file] [counts_file] [test_file] [output_file]
        Read in a gene tagged training input file and produce counts.
    """

if __name__ == "__main__":

    if len(sys.argv) != 5: # Expect exactly one argument: the training data file
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
    # Clean the infrequent words from data file 
    counter.compute_emission_prob()
    # Tag of words
    counter.tag_words1(testFile,outFile)

