from keras.preprocessing import sequence
import keras
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

class text_generator:
    def __init__(self,path_to_text):
        self.path_to_text = path_to_text
        self.checkpoints = './training_checkpoints'
        self.sequence_length = 100
        self.batch_size = 64
        self.embedding_dim = 256
        self.rnn_units = 1024
        self.buffer_size = 10000
        self.epochs = 10000

    def text_to_int(self,text,char2idx_):
        return np.array([char2idx_[c] for c in text])


    def split_input_target(self,chunk):  # example of "hannibal"
        input_text = chunk[:-1]  # hanniba
        target_text = chunk[1:]  # annibal
        return input_text, target_text  # hanniba, annibal

    def preprocess(self):
        text = Path(self.path_to_text).read_text()
        vocab = sorted(set(text)) #set of unique characters

        # Creating a dictionary from unique characters to indices
        char2idx = {u:i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        text_as_int = self.text_to_int(text,char2idx)


        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) # Create training examples
        sequences = char_dataset.batch(self.sequence_length+1, drop_remainder=True) #split into sequences and drop any remainder

        dataset = sequences.map(self.split_input_target)  # we use map to apply the this splitting to every target
        data = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)

        return data,len(vocab),char2idx,idx2char

    def loss(self,labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True) #custom loss function

    def build_model(self,vocab_size,batch_size=64):
        if (batch_size != self.batch_size) & (batch_size != 1):
            batch_size = self.batch_size
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, self.embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(self.rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
      ])
        model.compile(optimizer='adam', loss=self.loss)
        return model


    def train_model(self,data,vocab_size):
        checkpoint_prefix = os.path.join(self.checkpoints, "ckpt_{epoch}")

        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True) #only save weights to filepath
        class haltCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if(logs.get('loss') <= 0.5):
                    print("\n\n\nReached 0.5 loss value so cancelling training!\n\n\n")
                    self.model.stop_training = True
        loss_callback = haltCallback()


        model_trained = self.build_model(vocab_size).fit(data,epochs=self.epochs,callbacks=[checkpoint_callback,loss_callback])

    def build_generator_model(self,vocab_size):
        model = self.build_model(vocab_size,batch_size=1)
        model.load_weights(tf.train.latest_checkpoint(self.checkpoints))
        model.build(tf.TensorShape([1,None]))
        return model

    def generate_text(self,vocab_size,start_string,num_generate,char2idx,idx2char,predictability=1):
        model = self.build_generator_model(vocab_size)


          # oreorocess our string
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension

            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / predictability
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return (start_string + ''.join(text_generated))

    def main(self,start_string,predictability,num_generate):
        data,vocab_size,char2idx,idx2char = self.preprocess()
        self.train_model(data,vocab_size)
        generated_text = self.generate_text(vocab_size,start_string,num_generate,char2idx,idx2char,predictability
                )
        return generated_text
