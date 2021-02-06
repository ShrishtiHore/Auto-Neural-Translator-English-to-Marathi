# Neural Translator English to Marathi

Machine translation refers to translating phrases across languages using deep learning and specifically with RNN (Recurrent Neural Nets). Most of these are complex system that is they are combined system of various algorithms. But, at its core, NMT uses sequence-to-sequence ( seq2seq ) RNN cells. Such models could be character level but word level models remain common. 

-----

**Code and Resources:** 

**Language:** Python 3.8

**Libraries and Modules:** numpy, tensorflow, keras and pandas

**Dataset:** [Marathi and English Phrases Dataset](http://www.manythings.org/) 

![nmet](https://github.com/ShrishtiHore/Neural-Translator-English-to-Marathi/blob/main/nmt-model-fast.gif)

**Step 1: Downloading Datset and Libraries**
**(a) Importing the Libraries**
Import Tensorflow and Keras. From Keras, import modules that build NN layers, preprocess data and construct LSTM models.

**(b) Reading the Data**
The given dataset contains more than 30k pairs of English-Marathi phrases.Just donwload the dataset and unzip it and read it using Pandas.

**Step 2: Preparing input data for the Encoder**
The Encoder model will be fed input data which are preprocessed English sentences. The preprocessing is done as follows:

1.   Tokenizing the English sentences from eng_lines.
2.   Determining the maximum length of the English sentence that's max_input_length.
3.   Padding the tokenized_eng_lines to the max_input_length.
4.   Determining the vocabulary size ( num_eng_tokens ) for English words.

**Step 3: Preparing Input Data for the Decoder (decoder_input_data)**

The decoder model will be fed the preprocessed Marathi lines. The preprocessing steps are similar to the ones which are above. This one step is carried out before the other steps.

*  Append <START> tag at the first position in each Marathi sentence.
*  Append <END> tag at the last position in each Marathi sentence.

**Step 4: Preparing Target Data for the Decoder (decoder_target_data)
We take a copy of tokenized_mar_lines and modify it like this.**

1. We remove the <start> tag which we appended earlier. Hence, the word ( which is <start> in this case ) will be removed.
2. Convert the padded_mar_lines ( ones which do not have <start> tag ) to one-hot vectors.

`So [ '<start>' , 'hello' , 'world' , '<end>' ]`
will become:
` [ 'hello' , 'world' , '<end>' ]`

**Step 5: Defining the Encoder-Decoder Model**

The model will have Embedding, LSTM and Dense layers. The basic configuration is as follows.
*   2 Input Layers : One for encoder_input_data and another for decoder_input_data.
*   Embedding layer : For converting token vectors to fix sized dense vectors. ( Note :  Don't forget the mask_zero=True argument here )
*   LSTM layer : Provide access to Long-Short Term cells.

Working : 

1.   The encoder_input_data comes in the Embedding layer (  encoder_embedding ). 
2.   The output of the Embedding layer goes to the LSTM cell which produces 2 state vectors ( h and c which are encoder_states )
3.   These states are set in the LSTM cell of the decoder.
4.   The decoder_input_data comes in through the Embedding layer.
5.   The Embeddings goes in LSTM cell ( which had the states ) to produce sequences.

**Step 6: Train the Model**
We train the model for a number of epochs with RMSprop optimizer and categorical crossentropy loss function.

**Step 7: Defining Inference Models**
We create inference models which help in predicting translations.

* Encoder inference model : Takes the English sentence as input and outputs LSTM states ( h and c ).

* Decoder inference model : Takes in 2 inputs, one are the LSTM states ( Output of encoder model ), second are the Marathi input seqeunces ( ones not having the <start> tag ). It will output the translations of the English sentence which we fed to the encoder model and its state values.

**Step 8: Making some translations**
1.   First, we take a English sequence and predict the state values using enc_model.
2.   We set the state values in the decoder's LSTM.
3.   Then, we generate a sequence which contains the <start> element.
4.   We input this sequence in the dec_model.
5.   We replace the <start> element with the element which was predicted by the dec_model and update the state values.
6.   We carry out the above steps iteratively till we hit the <end> tag or the maximum sequence length.
  
**Result**

![result](https://github.com/ShrishtiHore/Neural-Translator-English-to-Marathi/blob/main/res.PNG)

**References**

1. https://www.analyticsvidhya.com/blog/2019/01/neural-machine-translation-keras/
2. https://medium.com/predict/creating-a-chatbot-from-scratch-using-keras-and-tensorflow-59e8fc76be79
3. http://www.manythings.org/
4. https://www.analyticsvidhya.com/blog/2020/01/3-important-nlp-libraries-indian-languages-python/
