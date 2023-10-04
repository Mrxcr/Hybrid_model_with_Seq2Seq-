import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
pio.renderers.default = 'vscode'
from tqdm import tqdm
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Add
from tensorflow.keras import regularizers, optimizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import load_model


class Seq2Seq_conventional:
    '''
    Conventional Seq2Seq network
    '''
    def __init__(self, latent_dim, timestep_encoder, timestep_decoder, Dim_encoder_In, Dim_decoder_In, Dim_decoder_out) -> None:
        self.latent_dim = latent_dim # number of hidden neurons in LSTM cells
        self.timestep_encoder = timestep_encoder # number of timsteps for encoder
        self.timestep_decoder = timestep_decoder # number of timsteps for decoder
        self.Dim_encoder_In = Dim_encoder_In # Input of encoder
        self.Dim_decoder_In = Dim_decoder_In # Input of decoder, incorporate ture values in the last step
        self.Dim_decoder_out = Dim_decoder_out # Ouput of decoder

        self.model_training = self.seq2seq_model_builder()

    def seq2seq_model_builder(self):
        '''
        Build a Seq2Seq network
        '''
        self.encoder_inputs = Input(shape=(self.timestep_encoder, self.Dim_encoder_In), name="encoder_input")
        encoder = LSTM(self.latent_dim, activation="tanh", return_state=True, name="encoder")
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
        self.encoder_states = [state_h, state_c]

        self.decoder_inputs = Input(shape=(self.timestep_decoder, self.Dim_decoder_In), name="decoder_input")
        self.decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name="decoder", activation="tanh")
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,
                                            initial_state=self.encoder_states)
        self.decoder_Hiddendense1 = Dense(32, name="Hiddendense1", activation="tanh")              
        self.decoder_Hiddendense2 = Dense(16, name="Hiddendense2", activation="tanh")                           
        self.decoder_dense = Dense(self.Dim_decoder_out, name="Output_Dense")
        decoder_outputs = self.decoder_Hiddendense1(decoder_outputs)
        decoder_outputs = self.decoder_Hiddendense2(decoder_outputs)
        decoder_outputs = self.decoder_dense(decoder_outputs)

        model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
        return model

    def train(self, Train_encoder_input, Train_decoder_input, Train_decoder_output
                  , Val_encoder_input, Val_decoder_input, Val_decoder_output
                  , batch_size=100, epochs=300, patience=50):
        '''
        Train the Seq2Seq network using the teacher forcing method
        '''
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, restore_best_weights=True)
        # optimizer = optimizers.Adam(learning_rate=0.01)
        self.model_training.compile(optimizer="Adam", loss='mse')
        self.Dim_decoder_Inhistory = self.model_training.fit([Train_encoder_input, Train_decoder_input], Train_decoder_output,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([Val_encoder_input, Val_decoder_input], Val_decoder_output),
                callbacks=[es],
                verbose=2)

    def build_model_inference(self):
        '''
        Restructure the Seq2Seq for inferencing
        '''
        self.encoder_model_inf = Model(self.encoder_inputs, self.encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_inputs_inf = Input(shape=(1, self.Dim_decoder_In))
        decoder_outputs, state_h, state_c = self.decoder_lstm(decoder_inputs_inf, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_Hiddendense1(decoder_outputs)
        decoder_outputs = self.decoder_Hiddendense2(decoder_outputs)
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.decoder_model_inf = Model(
            [decoder_inputs_inf] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    def predict(self, encoder_input, decoder_input_inf):
        '''
        Inference, i.e., predict current output using the output in the last step
        decoder_input_inf is made of exgenuous inputs
        '''
        self.build_model_inference()

        num_sam = encoder_input.shape[0]

        states_value = self.encoder_model_inf.predict(encoder_input, verbose=0)
        # Initial decoder input: The last TR in encoder_input + exogenous inputs [Fc Tcin BZ]        
        decoder_input_step = np.concatenate([np.zeros((num_sam, 1, 7)), decoder_input_inf[:, 0, :][:, np.newaxis, :]], axis=2)

        # Inference residuals step by step 
        list_decoder_output = []
        for step in range(self.timestep_decoder):
            output_tokens, h, c = self.decoder_model_inf.predict(
                [decoder_input_step] + states_value, verbose=0)

            list_decoder_output.append(output_tokens)

            if step + 1 == self.timestep_decoder:
                break

            # Update decoder_input_step and states
            decoder_input_step = np.concatenate([output_tokens, decoder_input_inf[:, step + 1, :][:, np.newaxis, :]], axis=2)
            states_value = [h, c]
        return np.concatenate(list_decoder_output, axis=1)
    


class Seq2Seq_hybrid:
    '''
    Seq2Seq network for hybrid modeling
    '''
    def __init__(self, latent_dim, timestep_encoder, timestep_decoder, Dim_encoder_In, Dim_decoder_In, Dim_decoder_out) -> None:
        self.latent_dim = latent_dim
        self.timestep_encoder = timestep_encoder
        self.timestep_decoder = timestep_decoder
        self.Dim_encoder_In = Dim_encoder_In
        self.Dim_decoder_In = Dim_decoder_In # made up by nominal predictions by UKF-FP and exgenuous inputs
        self.Dim_decoder_out = Dim_decoder_out

        self.model_training = self.seq2seq_model_builder()

    def seq2seq_model_builder(self):
        # Build model for training 
        self.encoder_inputs = Input(shape=(self.timestep_encoder, self.Dim_encoder_In), name="encoder_input")
        encoder = LSTM(self.latent_dim, activation="tanh", return_state=True, name="encoder")
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
        self.encoder_states = [state_h, state_c]

        self.decoder_inputs = Input(shape=(self.timestep_decoder, self.Dim_decoder_In), name="decoder_input")
        self.decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name="decoder", activation="tanh")
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,
                                            initial_state=self.encoder_states)
        self.decoder_Hiddendense1 = Dense(32, name="Hiddendense1"
                                                                # , kernel_regularizer=regularizers.L1L2(l2=1e-5)
                                                                #,  bias_regularizer=regularizers.L2(1e-4)
                                                                , activation="tanh"
                                                                )              
        self.decoder_Hiddendense2 = Dense(32, name="Hiddendense2"
                                                                # , kernel_regularizer=regularizers.L1L2(l2=1e-5)
                                                                #,  bias_regularizer=regularizers.L2(1e-4)
                                                                , activation="tanh"
                                                                )                           
        self.decoder_dense = Dense(self.Dim_decoder_out, name="Output_Dense")
        self.concat = Concatenate(axis=-1, name="resNet_Add")
        # Skip connections
        concat_outputs = self.concat([decoder_outputs, self.decoder_inputs[:]])
        decoder_outputs = self.decoder_Hiddendense1(concat_outputs)
        decoder_outputs = self.decoder_Hiddendense2(decoder_outputs)
        decoder_outputs = self.decoder_dense(decoder_outputs)

        model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
        return model

    def train(self, Train_encoder_input, Train_decoder_input, Train_decoder_output
                  , Val_encoder_input, Val_decoder_input, Val_decoder_output
                  , batch_size=100, epochs=300, patience=50):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, restore_best_weights=True)
        optimizer = optimizers.Adam(learning_rate=0.001)
        self.model_training.compile(optimizer=optimizer, loss='mse')
        self.Dim_decoder_Inhistory = self.model_training.fit([Train_encoder_input, Train_decoder_input], Train_decoder_output,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([Val_encoder_input, Val_decoder_input], Val_decoder_output),
                callbacks=[es],
                verbose=2)

    def pred(self, encoder_inputs, decoder_inputs):
        return self.model_training.predict([encoder_inputs, decoder_inputs])