#The program will take time to run, so be patient :)
#when setting up the model for the first time, uncomment the creating model part and comment the loading model part
import sklearn
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Activation
from tensorflow.keras.optimizers import Adam

#Load the file
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
#Read the file in binary mode
#Make the characters lowercase to increase accuracy
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()



text = text[300000:800000] #Take a part of the text to reduce the training time

characters = sorted(set(text)) #Get the unique characters
char_to_idx = {u:i for i, u in enumerate(characters)} #Create a dictionary to map the characters to indices
idx_to_char = {i:u for i, u in enumerate(characters)} #Create a dictionary to map the indices to characters


sequence_length = 50 #The length of the sequence
step = 3 #The step size, which is how many characters we skip before creating a new sequence
'''''
#Create Model
sentence = []
next_char = []


for i in range(0, len(text) - sequence_length, step): #Create the sequences
    sentence.append(text[i: i + sequence_length]) #Add the sequence to the list
    next_char.append(text[i + sequence_length]) #Add the next character to the list

x = np.zeros((len(sentence), sequence_length, len(characters)), dtype=np.bool) #Create the input data, sets all the values to 0 except for the characters in the sequence
y = np.zeros((len(sentence), len(characters)), dtype=np.bool) #Create the output data, sets all the values to 0 except for the next character

for i, sentence in enumerate(sentence): #Loop through the sequences
    for t, char in enumerate(sentence): #Loop through the characters in the sequence
        x[i, t, char_to_idx[char]] = 1 #Set the value to 1 when a character is found
    y[i, char_to_idx[next_char[i]]] = 1 #Set the value to 1 when the next character is found
'''
# model = Sequential() #Create the model
#
# model.add(LSTM(128, input_shape=(sequence_length, len(characters)))) #Add the LSTM layer
# model.add(Dense(len(characters))) #Add the dense layer
# model.add(Activation('softmax')) #Add the activation layer
#
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001)) #Compile the model
#
# model.fit(x, y, batch_size=256, epochs=4) #Train the model
#
# model.save('myModel.model') #Save the model

#Load the model

model = tf.keras.models.load_model('myModel.model') #Load the model

#Higher temperature means more random
def sample(preds, temperature=1.0): #Function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64') #Convert the predictions to a numpy array
    preds = np.log(preds) / temperature #Apply the temperature
    exp_preds = np.exp(preds) #Apply the exponential function
    preds = exp_preds / np.sum(exp_preds) #Get the probabilities
    probas = np.random.multinomial(1, preds, 1) #Sample the index
    return np.argmax(probas) #Return the index

def generate_text(legnth,temperature):
    start_index = random.randint(0, len(text) - sequence_length - 1) #Get a random index
    generated = '' #Create an empty string
    sentence = text[start_index: start_index + sequence_length] #Get the first sentence
    generated += sentence #Add the sentence to the string

    for i in range(legnth): #Loop through the length
        x_pred = np.zeros((1, sequence_length, len(characters))) #Create the input data
        for t, char in enumerate(sentence): #Loop through the characters in the sequence
            x_pred[0, t, char_to_idx[char]] = 1 #Set the value to 1 when a character is found

        preds = model.predict(x_pred, verbose=0)[0] #Get the predictions
        next_index = sample(preds, temperature) #Get the next index
        next_char = idx_to_char[next_index] #Get the next character

        generated += next_char #Add the character to the string
        sentence = sentence[1:] + next_char #Add the character to the sequence

    return generated #Return the string
print(generate_text(1000,0.3)) #Generate the text
