## Part 1

#### Unigram Answers
- "I'm not ready to go," said the the the the the the the the the the the the the the the the the the the the the the the the the
- Lily and Max were best friends. One day the the the the the the the the the the the the the the the the the the the the the the the the the
- He picked up the juice and the the the the the the the the the the the the the the the the the the the the the the the the the
- It was raining, so the the the the the the the the the the the the the the the the the the the the the the the the the
- The end of the story was the the the the the the the the the the the the the the the the the the the the the the the the the
#### 5-Gram Answers
- "I'm not read to go," said, "i wanted to the boy who listen the boy who listen the boy who listen the boy who listen the boy who listen the boy w
- Lily and Max were best friends. One day, the boy wo listen the boy who listen the boy who listen the boy who listen the boy who listen the
- He picked up the juice and said, "i wanted to the boy who listen the boy who listen the boy who listen the boy who listen the
- It was raining, so happ and said, "i wanted the boy who listen the boy who listen the boy who listen the boy who l
- The end of the story was a little girl named lily was a little girl named lily was a little girl named lily was a little gir

The 5-gram is certainly better. It at least repeats a phrase instead of falling into "the" over and over. Both models are lacking any form of deeper context that will allow it to avoid these patterns, and could use a larger training data set to create a more diverse training environment where these patters are less likely.

## Part 2 / 3

#### Vanilla RNN
- "I'm not ready to go," said and said, "i want to play with her mom and said, "i wondered the boy named timmy.<EOS> she was a little
- Lily and Max were best friends. One day was so happy and said, "i want to play with her mom and said, "i wondered the boy named timmy.<EOS> she
- He picked up the juice ande and said, "i want to play with her mom and said, "i wondered the boy named timmy.<EOS> she was a littl
- It was raining, son and said, "i want to play with her mom and said, "i wondered the boy named timmy.<EOS> she was a littl
- The end of the story was and said, "i want to play with her mom and said, "i wondered the boy named timmy.<EOS> she was a little
#### LSTM RNN
- "I'm not ready to go," saiden and said, "thank you, lily.<EOS> tom said, "thank you, lily.<EOS> tom said, "thank you, lily.<EOS> tom said,
- Lily and Max were best friends. One day, they were happy.<EOS> the started tom was so happy.<EOS> the started tom was so happy.<EOS> the started tom wa
- He picked up the juice andy and they were happy.<EOS> the started tom was so happy.<EOS> the started tom was so happy.<EOS> the started to
- It was raining, sone the park.<EOS>" the store was so happy.<EOS> the started tom was so happy.<EOS> the started tom was so happy.
- The end of the story was the started to the park.<EOS>" the store was so happy.<EOS> the started tom was so happy.<EOS> the started tom

a. In the examples written here, the models are somewhat similar in their coherence. However, throughout testing / running the program I generally found the LSTM model to have more variety and therefore give the appearance of more coherence.
b. These RNN models, even when they get repetitive, repeat over a greater length than the n-gram models. The unigram model for example, is likely to form extremely short loops, and the 5-gram model just extends this slightly. The RNN models also appear to repeat more of a sentence fragment compared to the word loops of the n-gram models, although this may just be because the fragments are longer and allow for it.
c. These models could also use a better, deeper understanding of context that goes beyond just what words have been written where, but *why* they were. For example, "the started tom was so happy" doesn't make sense, and if the model understood that it could avoid writing that fragment.
