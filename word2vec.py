import numpy as np

corpus = ["natural", "language", "process", "and", "machine", "learning", "is", "fun", "and", "exciting"]

settings = {
    #上下文单词是与目标单词相邻的单词
    "window_size": 2,
    #这是单词嵌入(word embedding)的维度，通常其的大小通常从100到300不等，取决于词汇库的大小超过300维度会导致效益递减
    "n": 10,
    "epochs":50,
    "learning_rate":0.01
}

class word2vec():
    def __init__(self):
        self.n = settings["n"]
        self.lr = settings["learning_rate"]
        self.epochs = settings["epochs"]
        self.window = settings["window_size"]
        
    def generate_training_data(self, settings, corpus):
        word_counts = defaultdict(int)
        # 统计单词出现的
        for row in corpus:
            for word in row:
                word_counts[word] += 1
                
        self.v_count = len(word_counts.keys())
        self.words_list = list(word_counts.keys())
        
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
        
        training_data = []
        # get the sentence in corpus
        for sentence in corpus:
            # get length of sentence
            sent_len = len(sentence)
            # get the word of sentence
            for i, word in enumerate(sentence):
                w_target = self.word2onehot(sentence[i])
                # Cycle through context window
                w_context = []
                # Note window_size 2 will have range of 5 values
                for j in range(i - self.window, i + self.window+1):
                    # Criteria fot context word
                    # target word cannot be context word (j != i)
                    # index must be greater or equal than 0 (j >= 0)
                    # index must be less or equal than length of sentence (j <= sent_len-1)
                    if j != i and j <= sent_len-1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                        # print(sentence[i], sentence[j])
                    training_data.append([w_target, w_context])
        return np.array(training_data)
