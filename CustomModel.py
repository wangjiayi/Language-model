import math, collections

class CustomModel:
  def __init__(self, corpus):
    """Initial custom language model and structures needed by this mode"""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.bigramCounts = collections.defaultdict(lambda: 0)

    self.counts_a = collections.defaultdict(lambda: 0)
    self.counts_b = collections.defaultdict(lambda: 0)


    self.train(corpus)
  def train(self, corpus):
    """ Takes a corpus and trains your language model.
    """  
    # TODO your code here
    for sentence in corpus.corpus:
      # here for unigram
      for datm in sentence.data:
        self.unigramCounts[datm.word] = self.unigramCounts[datm.word] + 1
        self.total = self.total + 1
        # here for bigram
      for index in range(0, len(sentence.data) - 1):
        first_word = sentence.data[index].word
        second_word = sentence.data[index + 1].word
        tok = '%s %s' % (first_word, second_word)
        self.bigramCounts[tok] = self.bigramCounts[tok] + 1
        # here for unigram
    self.unigramCounts['UNK'] = 0
    for index in self.unigramCounts:
      self.unigramCounts[index] = self.unigramCounts[index] + 1
      self.total = self.total + 1
      # here for bigram
    for uni in self.unigramCounts:
      for bi in self.bigramCounts:
        if bi.startswith(uni):
          self.counts_a[uni] = self.counts_a[uni] + 1
      for bi in self.bigramCounts:
        if bi.endswith(uni):
          self.counts_b[uni] = self.counts_b[uni] + 1

  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    # TODO your code here
    ret_score = 0.0
    for index in range(0, len(sentence) - 1):
      if (self.unigramCounts[sentence[index]] > 0):
        first_tok = sentence[index]
      second_tok = sentence[index + 1]
      bigram_tok = '%s %s' % (first_tok, second_tok)
      bigram_counts = self.bigramCounts[bigram_tok]

      top = max(bigram_counts - 1.5, 0)
      bottom = self.unigramCounts[first_tok]
      dis_bigram = top / bottom

      lemba = 1.5 / bottom * (self.counts_a[first_tok])
      continytion_prob = (self.counts_b[second_tok]) / (len(self.bigramCounts) * 2.0)
      prob = dis_bigram + lemba * continytion_prob

      ret_score = ret_score + math.log(prob + .0000001)
    return ret_score



