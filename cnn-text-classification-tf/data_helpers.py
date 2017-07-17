import numpy as np
import itertools
from collections import Counter



'''
此函数的作用是(文本，标签)对
'''
def load_data_and_labels(positive_data_file, negative_data_file):
    
    positive_examples = list(open(positive_data_file, "r",encoding='UTF-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
   
    negative_examples = list(open(negative_data_file, "r",encoding='UTF-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

'''
此函数的作用是返回num_epochs*num_batches_per_epoch个batch作为训练数据
'''
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1  
    print(num_batches_per_epoch)
    for epoch in range(num_epochs):  
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices] #打乱后的数据
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


