import re
import ast
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from wordenums import UNK, PAD, NUM


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.path = './data/qa_Software.json'

        self.questions, self.question_types = self.get_preprocessed_data()

        # tokenized documents
        self.documents = [question.split(" ") for question in self.questions]

        # list of unique words in the dataset
        self.unique_words = self.get_unique_words()

        # list of unique label classes
        self.unique_labels = list(set(self.question_types))

        # word mappings
        self.index_to_word = {index: word for index, word in enumerate(self.unique_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.unique_words)}

        # target mappings
        self.index_to_label = {index: label for index, label in enumerate(self.unique_labels)}
        self.label_to_index = {label: index for index, label in enumerate(self.unique_labels)}

        # input sequences -> indexed tokenized documents
        self.input_sequences = [[self.word_to_index[word] for word in doc] for doc in self.documents]

        # targets
        self.targets = [self.label_to_index[qt] for qt in self.question_types]

    def get_preprocessed_data(self):
        with open(self.path) as f:
            rows = []
            for line in f.readlines():
                try:
                    rows.append(ast.literal_eval(line.strip()))
                except:
                    print("ERROR: unable to load json for entry: {}".format(line))

        dataset = np.array([[row['asin'], row['questionType'], row['question']] for row in rows])

        questions = []
        question_types = []
        for row in dataset:
            # Preprocess questions
            inp = self.preprocess_text(row[2])

            # filter out questions that are less than 3 words long
            if len(inp.split(" ")) >= 3:
                questions.append(inp)
                question_types.append(row[1].lower())

        return questions, question_types

    def preprocess_text(self, text):
        text = text.strip()

        # convert to lowercase
        text = text.lower()

        # remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))

        # represent numbers with '<num>'
        text = re.sub(r"\d+(.\d+)?", NUM, text)

        return text

    def get_unique_words(self):
        words = [PAD, NUM, UNK]
        for text in self.documents:
            for word in text:
                if word not in words:
                    words.append(word)
        return words

    def pad_input_sequences(self):
        d_len = [len(inp) for inp in self.input_sequences]
        d_len_max = max(d_len)

        padded_sequences = []
        for inp in self.input_sequences:
            inp_len = len(inp)
            if inp_len != d_len_max:
                pad_index = self.word_to_index[PAD]
                padded_sequences.append([pad_index] * (d_len_max - inp_len) + inp)
            else:
                padded_sequences.append(inp)

        return padded_sequences

    #
    def get_train_val_test_split(self, X, y, train_val_split, val_test_split):
        X_train, X_testing, y_train, y_testing = train_test_split(X, y, test_size=train_val_split, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_testing, y_testing, test_size=val_test_split, random_state=42)
        return [X_train, y_train], [X_val, y_val], [X_test, y_test]

    #
    def print_documents_length_info(self):
        input_Sequence_lengths = [len(inp) for inp in self.input_sequences]
        shortest_seq_length = min(input_Sequence_lengths)
        longest_seq_length = max(input_Sequence_lengths)
        print("Longest Sequence (# of words): {}".format(longest_seq_length))
        print("Shortest Sequence (# of words): {}".format(shortest_seq_length))
        pd.Series(input_Sequence_lengths).hist()
        plt.show()
        print(pd.Series(input_Sequence_lengths).describe())


