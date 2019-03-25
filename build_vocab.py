import nltk
import pickle
import argparse
from collections import Counter


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(path, threshold):
    """Build a simple vocabulary wrapper."""
    dataset = ['train', 'val', 'test']
    # dataset = ['train']
    counter = Counter()
    
    for i in dataset:
        data_path = path + i + '_split.pkl'
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        ids = data.keys()
        for j, id in enumerate(ids):
            ingredient = data[id]['ingredients']
            for k in range(len(ingredient)):
                # tokens = nltk.tokenize.word_tokenize(ingredient[k].lower())
                tokens = [ingredient[k].lower()]
                counter.update(tokens)
            
            if (j+1) % 1000 == 0:
                print("[{}/{}] Tokenized the ingredients.".format(j+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    vocab.add_word(',')

    for i, word in enumerate(words):
        vocab.add_word(word)

    import pdb; pdb.set_trace()
    return vocab

def main(args):
    vocab = build_vocab(path=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='../im2recipe-Pytorch/data/recipe1M/', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/ingredient_vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)