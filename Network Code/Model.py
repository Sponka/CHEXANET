#!/usr/bin/env python
# coding: utf-8

# In[5]:


import gc
import os
import glob
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from platform import python_version
from keras.models import Sequential
from wandb.keras import WandbCallback
from keras.layers import Dense, Reshape
from wandb.keras import WandbMetricsLogger
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout, PReLU, ELU
from tensorflow.keras.layers import Concatenate,ZeroPadding2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import Multiply, add, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import Input, Lambda, Activation, Add, multiply, add, concatenate
from keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Conv2DTranspose, Flatten, Dense, Reshape

# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--loss_name', type=str, default='MAE', help='Loss name')
parser.add_argument('--loss', type=str, default=None, help='loss')
parser.add_argument('--bs', type=int, default=16, help='Batch size')
parser.add_argument('--spe', type=int, default=500, help='Steps per epoch')
args = parser.parse_args()

loss_name =args.loss_name
batch_size = args.bs 
steps_per_epoch = args.spe
epoch_num = 100
learning_rate_value = 0.0001
activation = 'lrelu' # lrelu, prelu, elu
version_name = f'Model_{batch_size}-{steps_per_epoch}-{epoch_num}-{activation}-{loss_name}'
model_version = f'Unet_{batch_size}-{steps_per_epoch}-{epoch_num}-{activation}-{loss_name}'
# =============================================================================
    
def preprocessing_function(log_data, params=None):
    log_data = np.clip(log_data, a_min=-20, a_max=None)
    min_value = -20
    max_value = 0
    min_max_scaled_data = (log_data - min_value) / (max_value - min_value)*2 -1
    min_max_scaled_data = np.nan_to_num(min_max_scaled_data, 0)
    return min_max_scaled_data


def load_array_3d(filename):
    """Load numpy array"""
    with open(filename, 'rb') as f:
        arr = np.load(f)
    return arr


def data_norm(name, output = False, save_idx=None):
    """Load one data example and use log transform
    Return array of size (Layers, Molecules, 1)"""
    arr = load_array_3d(name) # arr shape is (111, 100) or (molecules, layers)
    if output :
        arr = arr[-1]
    if save_idx is not None:
        arr = arr[save_idx,:]
    arr = np.log10(arr)[:,:96] # logarithm 
    arr = arr.reshape((64,96,1))
    arr = preprocessing_function(arr) # logarithm 
    return arr


def load_data(data_dir, output=False, save_idx=None):
    """Load all data (Train/Test) and apply log transform
    Return it as np array of size (number_of_examples, Layers, Molecules,1)"""
    data_list = np.sort(glob.glob(f'{data_dir}/*npy'))
    array = []
    _ = [array.append(data_norm(name, output=output, save_idx=save_idx)) for name in data_list]
    array = np.array(array)
    return array, data_list


def split_data(input_data, output_data):
    """Split dataset into train/test/validation sets, assuming 80/10/10 split"""
    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data,
                                                        test_size=0.2, random_state=42)    
    X_test, X_val, y_test, y_val  = train_test_split(X_test, y_test, test_size=0.5,random_state=42) 
    return X_train, X_test, X_val, y_train, y_test, y_val


def load_names(X_dir):
    data_list_X = np.sort(glob.glob(f'{X_dir}/*npy'))
    file_names = [path.split('/')[-1] for path in data_list_X]
    X_train, X_test, X_val, _, _, _ = split_data(file_names, file_names)
    return X_train, X_test, X_val


class DataGenerator(Sequence):
    def __init__(self, filenames, batch_size, save_idx, configuration):
        self.filenames = filenames
        self.configuration = configuration
        self.batch_size = batch_size
        self.save_idx = save_idx
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.filenames)


    def __getitem__(self, idx):
        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = []
        batch_y = []
        batch_info =[]
        for filename in batch_filenames:
            input_data = data_norm(f'/../{filename}',  save_idx = self.save_idx)
            output_data = data_norm(f'/../{filename}', output=True, save_idx = self.save_idx)  # Update with your output data
            additional_info = self.configuration[self.configuration['System_ID'].isin([filename.split('.')[0]])]
            additional_info = list(additional_info[['planet_radius', 'planet_mass', 'co_ratio','isothermal_T', 'metalicity']].to_numpy()[0])
            batch_X.append(input_data)
            batch_y.append(output_data)
            batch_info.append(additional_info)
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)
        batch_info = np.array(batch_info)
        return [batch_X,batch_info],  batch_y

# Network
# =============================================================================
def activation_function(conv, act_type='lrelu', name='activation'):
    if act_type == 'lrelu':
        result = LeakyReLU(alpha=0.2,name=f'{act_type}_{name}')(conv)
        return result
        
    if act_type == 'prelu':
        result = PReLU(name=f'{act_type}_{name}')(conv)
        return result
        
    if act_type == 'elu':
        result = ELU(name=f'{act_type}_{name}')(conv)
        return result
    
    
def model_additional_info(input_dim=5, output_shape=(8,12,1)):
    """
    Creates a neural network model that takes a vector of specified input dimensions
    and outputs a map of specified shape.

    Parameters:
    input_dim (int): Number of variables in the input vector.
    output_shape (tuple): Dimensions of the output map.

    Returns:
    model: A Keras Sequential model.
    """

    from keras.models import Sequential
    from keras.layers import Dense, Reshape

    model = Sequential()
    model.add(Dense(32, activation='linear', input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))  # First layer
    model.add(Dense(64, activation='linear'))
    model.add(LeakyReLU(alpha=0.2))                       # Second layer
    model.add(Dense(output_shape[0] * output_shape[1], activation='linear'))
    model.add(LeakyReLU(alpha=0.2))  # Output layer
    model.add(Reshape(output_shape))                              # Reshaping output

    return model



def convolution_module(input_tensor, filters,num, drop_rate = 0.2):
    conv = Conv2D(filters, (3, 3), padding='same', name = f'convolution_{num}_2')(input_tensor)
    conv = LeakyReLU(alpha=0.2, name=f'lrelu_{num}')(conv)
    conv = Conv2D(filters, (3, 3), padding='same', name = f'convolution_{num}_2')(conv)
    conv = LeakyReLU(alpha=0.2, name=f'lrelu_{num}_2')(conv)
    pool = MaxPooling2D((2, 2), padding='same', name =f'pool{num}')(conv)
    return pool, conv


def AttnBlock2D(x, g, desired_dimensionality, num):
    x_shape = x.shape
    g_shape = g.shape
    xl = Conv2D(desired_dimensionality,(1,1),strides=(2,2),
                activation="relu",padding = "same", name=f'Attention_conv_x_{num}')(x)
    gl = Conv2D(desired_dimensionality,(1,1),
                activation="relu",padding = "same", name=f'Attention_conv_g_{num}')(g)
    xg = Add(name=f'Attention_add_{num}')([xl,gl])
    xg = Activation("relu", name=f'Attention_relu_{num}')(xg)
    xg = Conv2D(1,(1,1),activation="sigmoid",padding = "same", name=f'Attention_conv_{num}')(xg)
    xg_shape = xg.shape
    xg = UpSampling2D((x_shape[1]//xg_shape[1],x_shape[2]//xg_shape[2]), 
                                      name=f'Attention_up_{num}')(xg)
    output = Multiply(name=f'Attention_mult_{num}')([xg,x])
    return output


def attention_upsample_and_concat(x1, x2, output_channels, in_channels, filters,num,drop_rate = 0.2):
    pool_size = 2
    x2 = AttnBlock2D(x2, x1, in_channels, num)
    upsampled = Conv2DTranspose(output_channels, (pool_size, pool_size),
                                strides=(pool_size, pool_size), padding='same', name=f'upsample_{num}')(x1)
    concat = Concatenate(axis=3)([upsampled, x2])
    conv = Conv2D(filters, (3, 3), padding='same', name =f'convolution_{num}')(concat)
    conv = activation_function(conv, act_type=activation, name=f'{num}_up') 
    
    conv = Conv2D(filters, (3, 3), padding='same',name=f'convolution_{num}_2' )(conv)
    conv = activation_function(conv, act_type=activation,name= f'{num}_2_up')  
    return conv


def network(input, additional_input, additional_output_shape = (8, 12, 1)):
    pool1, conv1 = convolution_module(input, 32,1)
    pool2, conv2 = convolution_module(pool1, 64,2)
    pool3, conv3 = convolution_module(pool2, 128,3)
    pool4, conv4 = convolution_module(pool3, 256,4)

    additional_model = model_additional_info(output_shape = additional_output_shape)
    additional_output = additional_model(additional_input)
    
    combined = concatenate([conv4, additional_output])
    
    up7 = attention_upsample_and_concat(combined, conv3, 128, 257,128,7)
    up8 = attention_upsample_and_concat(up7, conv2, 64, 128,64,8)
    up9 = attention_upsample_and_concat(up8, conv1, 32, 64,32,9)
    conv10 = Conv2D(1, (1, 1), padding='same', name = 'convolution_10')(up9)
    return conv10
# =============================================================================


# Callbacks
# =============================================================================



class WandbCustomCallback(Callback):
    def __init__(self, val_generator, val_steps, molecules):
        super().__init__()
        self.val_generator = val_generator
        self.val_steps = val_steps  # This will be 1 if you want to use one batch
        self.molecules = molecules

    def on_epoch_end(self, epoch, logs=None):
        # Collect predictions and true labels from the validation generator
        # Since val_steps is 1, we only take one batch
        val_x, val_y = next(iter(self.val_generator))
        val_pred = self.model.predict(val_x)

        # Log the plots using the collected data
        self.log_prediction_plots(val_x,val_y, val_pred)
        self.log_prediction_plots_molecules(val_x,val_y, val_pred, self.molecules)
        self.height_abundance_plot(val_x,val_y, val_pred, self.molecules)

    def log_prediction_plots(self,val_x, true_data, pred_data):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
       
        ax.plot(true_data[:4,:,:].flatten(), pred_data[:4,:,:].flatten(), '.', color='royalblue')
        ax.plot([-1, 1], [-1, 1], '-', color='Black')
        plt.xlabel('Ground truth')
        plt.ylabel('Prediction')
        plt.title('All layers and molecules in 4 validation atmospheres')
        plt.grid()
        plt.tight_layout()
        wandb.log({"Predicted vs True - 4 Atmospheres": wandb.Image(plt)})
        plt.close()


    def log_prediction_plots_molecules(self,val_x, true_data, pred_data, molecules):
        molecules_list = [40, 43, 6, 45, 38, 13, 35, 53, 31, 46]
        for mol in molecules_list:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.plot(true_data[:4, mol, :].flatten(), pred_data[:4, mol, :].flatten(), '.', color='royalblue')
            ax.plot([-1, 1], [-1, 1], '-', color='Black')
            
            plt.xlabel('Ground truth')
            plt.ylabel('Prediction')
            plt.title(f'All layers for {molecules[mol]} in 4 validation atmospheres')
            plt.grid()
            plt.tight_layout()
            wandb.log({f"Predicted vs True - Molecule {molecules[mol]}": wandb.Image(plt)})
            plt.close()

    def height_abundance_plot(self,val_x, true_data, pred_data, molecules):
        molecules_list = ['CH3', 'CH4', 'O2', 'CO', 'CO2', 'C2H5', 'C2H2', 'C2H4', 'H2O']
        layers = np.arange(0, 64)
        colors = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", 
                  "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
        for chosen_atm in [0, 1, 2, 3]:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            for i, mol in enumerate(molecules_list):
                color = colors[i % len(colors)]
                
                ax.plot(self.inverse_preprocessing(true_data[chosen_atm, molecules.index(mol), :]),
                        layers, '-', label=mol, color=color)
                ax.plot(self.inverse_preprocessing(pred_data[chosen_atm, molecules.index(mol), :]),
                        layers, '--', label=f'{mol} P', color=color)
            plt.grid()
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Predicted (P, dashed line) vs Ground Truth (full line) abundances, atm {chosen_atm}')
            plt.xlabel('Abundance')
            plt.ylabel('Layer')
            plt.tight_layout()
            wandb.log({f"Abundance vs Layer - Atmosphere {chosen_atm}": wandb.Image(plt)})
            plt.close()

    def inverse_preprocessing(self, scaled_data):
        min_value = -20
        max_value = 0
        log_data = ((scaled_data + 1) / 2) * (max_value - min_value) + min_value
        return log_data

# =============================================================================


# Data Loading
# =============================================================================

configuration = pd.read_csv('../data/configuration/system_configuration.csv')

par_min = 0.4
par_max = 1.5
configuration['co_ratio']=(configuration['co_ratio']-par_min)/(par_max-par_min) * 2 - 1

par_min = 0.3370963034022992 
par_max = 3.876389678499647
configuration['planet_radius']=(configuration['planet_radius']-par_min)/(par_max-par_min) * 2 - 1

par_min = 0.3
par_max = 10
configuration['planet_mass']=(configuration['planet_mass']-par_min)/(par_max-par_min) * 2 - 1

par_min = 1100
par_max = 2000
configuration['isothermal_T']=(configuration['isothermal_T']-par_min)/(par_max-par_min) * 2 - 1

par_min = 0.5
par_max = 100
configuration['metalicity']=(configuration['metalicity']-par_min)/(par_max-par_min) * 2 - 1

save_molecules = np.array([ 34,  36,  39,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,
        51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
        64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,
        77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  90,  91,
        94,  96,  97, 100, 101, 102, 104, 105, 106, 108, 109, 110])

molecule_len = len(save_molecules)

molecules= ["HCNO","N2O","CH2CHO","CH3CO","NCO","CH3O","O2","CH3CHO","HNO","C",
            "CHCO","CO2H","HOCN","C2H5","C2H","CH2OH","CH","C2H6","C2H3","CH2CO","NNH",
            "H2CN","CH3OH","N4S","N2D","CN","1CH2","HNCO","NO","O3P","O1D","C2H4","NH",
            "3CH2","HCO","C2H2","H2CO","NH2","CO2","OH","CH3","HCN","NH3","CH4","N2",
            "CO","H2O","H","He","H2","N2H2","N2H3","HNOH","NH2OH","H2NO","C2N2","HCNH",
            "HNC","NCN","HCOH","HOCHO","H2Oc","CH4c","NH3c"]

X_train, X_test, X_val = load_names('/../data/ode_results_time_steps/')
train_generator = DataGenerator(X_train, batch_size, save_molecules,configuration)
test_generator = DataGenerator(X_test , batch_size, save_molecules,configuration)
val_generator = DataGenerator(X_val, batch_size, save_molecules,configuration)

# =============================================================================
            

input_shape = (64, 64, 1)
inputs = tf.keras.Input(shape=input_shape)
additional_input_shape = (5,)
additional_input = tf.keras.Input(shape=additional_input_shape)

output = network(inputs,additional_input)
model = Model(inputs=[inputs,additional_input], outputs=output)
wandb_custom_callback = WandbCustomCallback(val_generator, 2, molecules)

callbacks = [WandbMetricsLogger(), wandb_custom_callback,
    ModelCheckpoint(filepath=f'{version_name}/{model_version}.h5', monitor='val_loss',
                    save_best_only=True),
    TensorBoard(log_dir='logs'),
    EarlyStopping(monitor='val_loss', patience=7),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.2*0.2*learning_rate_value),
    CSVLogger(f'{version_name}/training_log.csv')]

model.compile(optimizer=Adam(learning_rate=learning_rate_value), loss='mean_absolute_error' )
model.fit(train_generator, epochs=epoch_num, validation_data=val_generator,callbacks=callbacks, steps_per_epoch=steps_per_epoch)


# In[2]:


input_shape = (64, 96, 1)
inputs = tf.keras.Input(shape=input_shape)
additional_input_shape = (5,)
additional_input = tf.keras.Input(shape=additional_input_shape)

output = network(inputs,additional_input)
model = Model(inputs=[inputs,additional_input], outputs=output)


# In[ ]:


model.summary()


# In[ ]:




