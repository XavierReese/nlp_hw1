import math
from collections import defaultdict
from utils import Vocab, read_data
"""You should not need any other imports."""

class NGramModel:
    def __init__(self, n, data):
        self.n = n
        self.vocab = Vocab()

        """TODO: Populate vocabulary with all possible characters/symbols in the data, including '<BOS>', '<EOS>', and '<UNK>'."""
        # Special Tokens
        for special in ["<BOS>", "<EOS>", "<UNK>"]:
            self.vocab.add(special)

        # Collect unique symbols from data
        for seq in data:
            for sym in seq:
                self.vocab.add(sym)

        self.counts = defaultdict(lambda: defaultdict(int))

    def start(self):
        return ['<BOS>'] * (self.n - 1) # Remember that read_data prepends one <BOS> tag. Depending on your implementation, you may need to remove or work around that. No n-gram should have exclusively <BOS> tags; initial context should be n-1 <BOS> tags and the first prediction should be of the first non-BOS token.

    def fit(self, data):
        """TODO: 
			* Train the model on the training data by populating the counts. 
				* For n>1, you will need to keep track of the context and keep updating it. 
				* Get the starting context with self.start().
		"""
        for seq in data:
            context = self.start()
            for sym in seq:
                context_tuple = tuple(context)
                self.counts[context_tuple][sym] += 1
                # print("CONTEXT: ", context_tuple, "SYM: ", sym, "COUNT: ", self.counts[context_tuple][sym])
                context += [sym]
                context = context[(len(context) - n):]
        self.probs = {}
        """TODO: Populate self.probs by converting counts to log probabilities with add-1 smoothing."""
        v = len(self.vocab)
        for ctx, targets in self.counts.items():
            total = sum(targets.values()) + v
            self.probs[ctx] = {}
            for symbol in self.vocab.sym2num:
                count = targets.get(symbol, 0)
                # if(count > 0): print("CTX: ", ctx, "SYMBOL: ", symbol, " COUNT: ", count)
                prob = (count + 1) / total
                self.probs[ctx][symbol] = math.log(prob)

    def step(self, context):
        """Returns the distribution over possible next symbols. For unseen contexts, backs off to unigram distribution."""
        context = self.start() + context
        context = tuple(context[(len(context) - n):])
        if context in self.probs:
            # print("CONTEXT in step(): ", context, "PROBS: ", self.probs[context])
            return self.probs[context]
        else:
            return {sym: math.log(1 / len(self.vocab)) for sym in self.vocab.sym2num}

    def predict(self, context):
        """TODO: Return the most likely next symbol given a context. Hint: use step()."""
        prob_dist = self.step(context)
        return max(prob_dist.items(), key=lambda x: x[1])[0]

    def evaluate(self, data):
        """TODO: Calculate and return the accuracy of predicting the next character given the original context over all sentences in the data. Remember to provide the self.start() context for n>1."""
        correct = 0
        total = 0
        for seq in data:
            context = self.start()
            for sym in seq:
                pred = self.predict(context)
                if pred == sym:
                    correct += 1
                total += 1
                context += [sym]
                context = context[(len(context) - n):]
        return correct / total if total > 0 else 0

if __name__ == '__main__':

    train_data = read_data('./data/train.txt')
    val_data = read_data('./data/val.txt')
    test_data = read_data('./data/test.txt')
    response_data = read_data('./data/response.txt')

    n = 1 # TODO: n=1 and n=5
    if(n > 1): n -= 1 # because for n=1 to be unigram, 5-gram must be n=4 based on HW definition
    model = NGramModel(n, train_data)
    model.fit(train_data)
    print(model.evaluate(val_data), model.evaluate(test_data))

    """Generate the next 100 characters for the free response questions."""
    for x in response_data:
        x = x[:-1] # remove EOS
        for _ in range(100):
            #print("PREDICT: ", x)
            y = model.predict(x)
            #print("PREDICTED: ", y)
            x += [y]
        print(''.join(x) + '\n')
