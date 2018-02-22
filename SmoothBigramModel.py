import math, collections

class SmoothBigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    # Tip: To get words from the corpus, try
    #    for sentence in corpus.corpus:
    #       for datum in sentence.data:  
    #         word = datum.word
    for sentence in corpus.corpus:
      for index in range(0, len(sentence.data) - 1):
        first_word = sentence.data[index].word
        second_word = sentence.data[index+1].word
        tok = '%s %s' % (first_word,second_word)
        self.bigramCounts[tok] = self.bigramCounts[tok] + 1
        self.total = self.total + 1
    self.bigramCounts['UNK'] = 0
    for index in self.bigramCounts:
      self.bigramCounts[index] = self.bigramCounts[index] + 1
      self.total = self.total + 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    ret_score = 0.0
    for index in range(0, len(sentence) - 1):
      first_tok = sentence[index]
      second_tok = sentence[index+1]
      tok = '%s %s' % (first_tok, second_tok)
      counts = self.bigramCounts[tok] + self.bigramCounts['UNK']
      if counts > 0:
        ret_score = ret_score + math.log(counts)
        ret_score = ret_score - math.log(self.total)
    return ret_score
