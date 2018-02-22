import math, collections

class SmoothUnigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
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
    self.unigramCounts['UNK'] = 0
    for sentence in corpus.corpus:
      for datm in sentence.data:
        self.unigramCounts[datm.word] = self.unigramCounts[datm.word] + 1
        self.total = self.total + 1
    for index in self.unigramCounts:
      self.unigramCounts[index] = self.unigramCounts[index] + 1
      self.total = self.total + 1


  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    ret_score = 0.0
    for index in sentence:
      counts = self.unigramCounts[index] + self.unigramCounts['UNK']
      if counts > 0:
        ret_score = ret_score + math.log(counts)
        ret_score = ret_score - math.log(self.total)
    return ret_score
