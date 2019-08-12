import numpy as np

from collections import defaultdict

corpus = [["natural", "language", "process", "and", "machine", "learning", "is", "fun", "and", "exciting"]]

# Word2Vec2模型有两个权重矩阵(w1和w2)，为了展示，
# 我们把值初始化到形状分别为(9x10)和(10x9)的矩阵。这便于反向传播误差的计算
getW1 = [[0.236, -0.962, 0.686, 0.785, -0.454, -0.833, -0.744, 0.677, -0.427, -0.066],
		[-0.907, 0.894, 0.225, 0.673, -0.579, -0.428, 0.685, 0.973, -0.070, -0.811],
		[-0.576, 0.658, -0.582, -0.112, 0.662, 0.051, -0.401, -0.921, -0.158, 0.529],
		[0.517, 0.436, 0.092, -0.835, -0.444, -0.905, 0.879, 0.303, 0.332, -0.275],
		[0.859, -0.890, 0.651, 0.185, -0.511, -0.456, 0.377, -0.274, 0.182, -0.237],
		[0.368, -0.867, -0.301, -0.222, 0.630, 0.808, 0.088, -0.902, -0.450, -0.408],
		[0.728, 0.277, 0.439, 0.138, -0.943, -0.409, 0.687, -0.215, -0.807, 0.612],
		[0.593, -0.699, 0.020, 0.142, -0.638, -0.633, 0.344, 0.868, 0.913, 0.429],
		[0.447, -0.810, -0.061, -0.495, 0.794, -0.064, -0.817, -0.408, -0.286, 0.149]]
 
getW2 = [[-0.868, -0.406, -0.288, -0.016, -0.560, 0.179, 0.099, 0.438, -0.551],
		[-0.395, 0.890, 0.685, -0.329, 0.218, -0.852, -0.919, 0.665, 0.968],
		[-0.128, 0.685, -0.828, 0.709, -0.420, 0.057, -0.212, 0.728, -0.690],
		[0.881, 0.238, 0.018, 0.622, 0.936, -0.442, 0.936, 0.586, -0.020],
		[-0.478, 0.240, 0.820, -0.731, 0.260, -0.989, -0.626, 0.796, -0.599],
		[0.679, 0.721, -0.111, 0.083, -0.738, 0.227, 0.560, 0.929, 0.017],
		[-0.690, 0.907, 0.464, -0.022, -0.005, -0.004, -0.425, 0.299, 0.757],
		[-0.054, 0.397, -0.017, -0.563, -0.551, 0.465, -0.596, -0.413, -0.395],
		[-0.838, 0.053, -0.160, -0.164, -0.671, 0.140, -0.149, 0.708, 0.425],
		[0.096, -0.995, -0.313, 0.881, -0.402, -0.631, -0.660, 0.184, 0.487]]

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
        # 统计单词出现的次数
        for row in corpus:
            for word in row:
                word_counts[word] += 1
        # 统计词典大小        
        self.v_count = len(word_counts.keys())
        # 将词典的key（单词）转换成list
        self.words_list = list(word_counts.keys())
        
        # 建立word为key，i为value的字典
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        # 建立i为key，word为value的字典
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
        
        # 存储训练数据
        training_data = []
        # get the sentence in corpus
        for sentence in corpus:
            # get length of sentence
            sent_len = len(sentence)
            # get the word of sentence
            for i, word in enumerate(sentence):
                # 取出sentence中的每个word，从one-hot转换成embedding，w_target存储中心词
                w_target = self.word2onehot(sentence[i])
                # Cycle through context window
                # w_context里面存储中心词上下文单词
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
                # w_target, w_context分别存储中心词和上下文单词
                training_data.append([w_target, w_context])
        return np.array(training_data)

    def word2onehot(self, word):
        # 构造一个全零list，取出对应word的index，置全零list的对应index为零
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec
    
    def train(self, training_data):
        self.w1 = np.array(getW1)
        self.w2 = np.array(getW2)
        # 随机初始化W1和W2
        # self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        # self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
        for i in range(self.epochs):
            self.loss = 0
            # w_t = vector for target word, w_c = vector for context words
            for w_t, w_c in training_data:
                y_pred, h, u = self.forward_pass(w_t)
        
      
    
    
    
    
    
    
    

