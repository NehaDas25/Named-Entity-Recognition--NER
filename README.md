# Work Report

## Information

- Name: <ins> DAS, NEHA </ins>
- GitHub: <ins> NehaDas25 </ins>


## Features

- Not Implemented:
  - what features have been implemented

<br><br>

- Implemented:
  - PART 1: EXPLPORING THE DATA
  - PART 1.2: DATA GENERATOR
    - Exercise 1: data_generator
      - Implemented a data_generator() function that takes in batch_size, x, y, pad, shuffle=False, verbose=False where x is a large list of sentences, and y is a list of the tags associated with those sentences and pad is a pad value.
      - Return a subset of those inputs in a tuple of two arrays (X,Y).
      - X and Y are arrays of dimension (batch_size, max_len), where max_len is the length of the longest sentence in that batch.
      - We will pad the X and Y examples with the pad argument. If shuffle=True, the data will be traversed in a random order.
      - Here within the while loop, two for loops are defined:
        1. The first stores temporal lists of the data samples to be included in the batch, and finds the maximum length of the sentences contained in it.
        2. The second one moves the elements from the temporal list into NumPy arrays pre-filled with pad values. 
      - The NumPy full function has been used here to fill the NumPy arrays with a pad value.
      - Tracking the current location in the incoming lists of sentences. Generators variables hold their values between invocations, so we create an index variable, initialize to zero, and increment by one for each sample included in a batch. However, we do not use the index to access the positions of the list of sentences directly. Instead, we use it to select one index from a list of indexes. In this way, we can change the order in which we traverse our original list, keeping untouched our original list
      - Since batch_size and the length of the input lists are not aligned, gathering a batch_size group of inputs may involve wrapping back to the beginning of the input loop. In our approach, it is just enough to reset the index to 0. We can re-shuffle the list of indexes to produce different batches each time.
      - This passed all the unit-test cases.

  - PART 2: BUILDING THE MODEL
    - Exercise 2: NER
      - Implemented the initialization step and the forward function of Named Entity Recognition system that is NER() function that takes in tags, vocab_size=35181, d_model=50.
      - To implement this model, google's trax package has been used, instead of implementing the LSTM from scratch and the necessary methods from a build in package has been provided in the assignment.
      - The following packages when constructing the model has been used here are:
        1. *tl.Serial()*: Combinator that applies layers serially.Here,the layers are passed as arguments to Serial, separated by commas like tl.Serial(tl.Embeddings(...), tl.Mean(...), tl.Dense(...), tl.LogSoftmax(...)).
        2. *tl.Embedding(vocab_size, d_feature)*: Initializes the embedding. In this case it is the size of the vocabulary by the dimension of the model.vocab_size is the number of unique words in the given vocabulary.d_feature is the number of elements in the word embedding (some choices for a word embedding size range from 150 to 300, for example).
        3. *tl.LSTM(n_units)*: This is Trax LSTM layer.Builds an LSTM layer with hidden state and cell sizes equal to n_units. In trax, n_units should be equal to the size of the embeddings d_feature.
        4. tl.Dense(): A dense layer. tl.Dense(n_units): the parameter n_units is the number of units chosen for this dense layer.
        5. tl.LogSoftmax(): Log of the output probabilities.
      - This passed all the unit-test cases as well.

  - PART 3: TRAINING
  - PART 3.1: Training the Model
    - Exercise 3: train_model
      - Implemented the train_model() to train the neural network that takes NER, train_generator, eval_generator, train_steps=1, output_dir='model' as inputs.
      - Created the TrainTask and EvalTask using your data generator.
      - **training task** that uses the train data generator defined in the cell above
        1. loss_layer = tl.CrossEntropyLoss()
        2. optimizer = trax.optimizers.Adam(0.01)
      - evaluation task that uses the validation data generator defined in the cell above and the following arguments
        1. metrics for EvalTask: tl.CrossEntropyLoss() and tl.Accuracy()
        2. In EvalTask set n_eval_batches=10 for better evaluation accuracy 
      - Created the trainer object by calling trax.supervised.training.Loop and passed in the following:
        1. model = NER
        2. train_task
        3. Eval_task
        4. output_dir = output_dir
      - This passed all the unit-test cases as well.

  - PART 4: COMPUTE ACCURACY
    - Exercise 4: evaluate_prediction
      - Implemented the evaluate_prediction() that takes pred, labels, pad, verbose=True as inputs.
      - Step 1: model(sentences) will give you the predicted output.
      - Step 2: Prediction will produce an output with an added dimension. For each sentence, for each word, there will be a vector of probabilities for each tag type. For each sentence,word, you need to pick the maximum valued tag. This will require np.argmax and careful use of the axis argument.
      - Step 3: Create a mask to prevent counting pad characters. It has the same dimension as output. 
      - Step 4: Compute the accuracy metric by comparing the outputs against the test labels. Take the sum of that and divide by the total number of unpadded tokens. Use the mask value to mask the padded tokens. Return the accuracy.
      - This passed all the unit-test cases as well.



        
<br><br>

- Partly implemented:
  - w3_unittest.py was also not implemented as part of assignment to pass all unit-tests for the graded functions().
  - utils.py which contains get_vocab(vocab_path, tags_path), get_params(vocab, tag_map, sentences_file, labels_file) has not been implemented, it was provided.

<br><br>

- Bugs
  - No Bugs

<br><br>


## Reflections

- Assignment is quiet knowledgeable. Gives a thorough understanding of the LSTM and its gates, Named Entity Recognition, train_task, eval_task, word padding.



## Output

### output:

<pre>
<br/><br/>
Out[2] - 

SENTENCE: Thousands of demonstrators have marched through London to protest the war in Iraq and demand the withdrawal of British troops from that country .

SENTENCE LABEL: O O O O O O B-geo O O O O O B-geo O O O O O B-gpe O O O O O

ORIGINAL DATA:
     Sentence #           Word  POS Tag
0  Sentence: 1      Thousands  NNS   O
1          NaN             of   IN   O
2          NaN  demonstrators  NNS   O
3          NaN           have  VBP   O
4          NaN        marched  VBN   O

Out[4] -

vocab["the"]: 9
padded token: 35180

Out[5] - 

{'O': 0, 'B-geo': 1, 'B-gpe': 2, 'B-per': 3, 'I-geo': 4, 'B-org': 5, 'I-org': 6, 'B-tim': 7, 'B-art': 8, 'I-art': 9, 'I-per': 10, 'I-gpe': 11, 'I-tim': 12, 'B-nat': 13, 'B-eve': 14, 'I-eve': 15, 'I-nat': 16}

Out[6] - 

The number of outputs is tag_map 17
Num of vocabulary words: 35181
The training size is 33570
The validation size is 7194
An example of the first sentence is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 9, 15, 1, 16, 17, 18, 19, 20, 21]
An example of its corresponding label is [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]

Out[8] - 

index= 5
index= 2
(5, 30) (5, 30) (5, 30) (5, 30)
[    0     1     2     3     4     5     6     7     8     9    10    11
    12    13    14     9    15     1    16    17    18    19    20    21
 35180 35180 35180 35180 35180 35180] 
 [    0     0     0     0     0     0     1     0     0     0     0     0
     1     0     0     0     0     0     2     0     0     0     0     0
 35180 35180 35180 35180 35180 35180]

Expected output:

index= 5
index= 2
(5, 30) (5, 30) (5, 30) (5, 30)
[    0     1     2     3     4     5     6     7     8     9    10    11
    12    13    14     9    15     1    16    17    18    19    20    21
 35180 35180 35180 35180 35180 35180] 
 [    0     0     0     0     0     0     1     0     0     0     0     0
     1     0     0     0     0     0     2     0     0     0     0     0
 35180 35180 35180 35180 35180 35180]

Out[9] - All tests passed

Out[11] -

Serial[
  Embedding_35181_50
  LSTM_50
  Dense_17
  LogSoftmax
]

Expected output:

Serial[
  Embedding_35181_50
  LSTM_50
  Dense_17
  LogSoftmax
]

Out[12] - All tests passed

Out[15] - All tests passed

Out[16] - 

Step      1: Total number of trainable weights: 1780117
Step      1: Ran 1 train steps in 0.96 secs
Step      1: train CrossEntropyLoss |  2.99960470
Step      1: eval  CrossEntropyLoss |  2.03760970
Step      1: eval          Accuracy |  0.02747587

Step    100: Ran 99 train steps in 16.52 secs
Step    100: train CrossEntropyLoss |  0.51940256
Step    100: eval  CrossEntropyLoss |  0.23112577
Step    100: eval          Accuracy |  0.94081764

Expected output (Approximately)

...
Step      1: Total number of trainable weights: 1780117
Step      1: Ran 1 train steps in 2.63 secs
Step      1: train CrossEntropyLoss |  4.49356890
Step      1: eval  CrossEntropyLoss |  3.41925483
Step      1: eval          Accuracy |  0.01685534

Step    100: Ran 99 train steps in 49.14 secs
Step    100: train CrossEntropyLoss |  0.61710459
Step    100: eval  CrossEntropyLoss |  0.27959008
Step    100: eval          Accuracy |  0.93171992


Out[18] - array([False,  True, False, False])

Out[19] - input shapes (7194, 70) (7194, 70)

Out[20] -

<class 'jaxlib.xla_extension.ArrayImpl'>
tmp_pred has shape: (7194, 70, 17)

Out[22] -

outputs shape: (7194, 70)
mask shape: (7194, 70) mask[0][20:30]: [ True  True  True False False False False False False False]
accuracy:  0.9543761

Expected output (Approximately)

outputs shape: (7194, 70)
mask shape: (7194, 70) mask[0][20:30]: [ True  True  True False False False False False False False]
accuracy:  0.9543761

Out[23] -

outputs shape: (3, 3)
mask shape: (3, 3) mask[0][20:30]: []
outputs shape: (3, 3)
mask shape: (3, 3) mask[0][20:30]: []
outputs shape: (3, 3)
mask shape: (3, 3) mask[0][20:30]: []
outputs shape: (3, 4)
mask shape: (3, 4) mask[0][20:30]: []
 All tests passed

Out[25] -

Peter B-per
Navarro, I-per
White B-org
House I-org
Sunday B-tim
morning I-tim
White B-org
House I-org
coronavirus B-tim
fall, B-tim

Expected Results

Peter B-per
Navarro, I-per
White B-org
House I-org
Sunday B-tim
morning I-tim
White B-org
House I-org
coronavirus B-tim
fall, B-tim

<br/><br/>
</pre>
