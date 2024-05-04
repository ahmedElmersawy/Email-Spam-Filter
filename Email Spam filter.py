import os
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
def load_data():
    print("Loading data...")
    
    ham_files_location = os.listdir('dataset/ham')
    spam_files_location = os.listdir('dataset/spam')
    data = []
    # Load ham email
    for file_path in ham_files_location:
        f = open('dataset/ham/' + file_path, "r")
        text = str(f.read())
        data.append([text, "ham"])
    
    # Load spam email
    for file_path in spam_files_location:
        f = open("dataset/spam/" + file_path, "r")
        text = str(f.read())
        data.append([text, "spam"])
    data = np.array(data)
    print("flag 1: loaded data")
    return data
# Preprocessing data: noise removal
def preprocess_data(data):
    print("Preprocessing data...")
    
    punc = string.punctuation           # Punctuation list
    sw = stopwords.words('english')     # Stopwords list
    for record in data:
        # Remove common punctuation and symbols
        for item in punc:
            record[0] = record[0].replace(item, "")
            # Lowercase all letters and remove stopwords 
        splittedWords = record[0].split()
        newText = ""
        for word in splittedWords:
            if word not in sw:
                word = word.lower()
                newText = newText + " " + word  
        record[0] = newText
        
    print("flag 2: preprocessed data")        
    return data
# Splitting original dataset into training dataset and test dataset
def split_data(data):
    print("Splitting data...")
    
    features = data[:, 0]   # array containing all email text bodies
    labels = data[:, 1]     # array containing corresponding labels
    print(labels)
    training_data, test_data, training_labels, test_labels =\
        train_test_split(features, labels, test_size = 0.27, random_state = 42)
    
    print("flag 3: splitted data")
    return training_data, test_data, training_labels, test_labels
def get_count(text):
    wordCounts = dict()
    for word in text.split():
        if word in wordCounts:
            wordCounts[word] += 1
        else:
            wordCounts[word] = 1
    
    return wordCounts
def euclidean_difference(test_WordCounts, training_WordCounts):
    total = 0
    for word in test_WordCounts:
        if word in test_WordCounts and word in training_WordCounts:
            total += (test_WordCounts[word] - training_WordCounts[word])**2
            del training_WordCounts[word]
        else:
            total += test_WordCounts[word]**2
    for word in training_WordCounts:
            total += training_WordCounts[word]**2
    return total**0.5
def get_class(selected_Kvalues):
    spam_count = 0
    ham_count = 0
    for value in selected_Kvalues:
        if value[0] == "spam":
            spam_count += 1
        else:
            ham_count += 1
    if spam_count > ham_count:
        return "spam"
    else:
        return "ham"
def knn_classifier(training_data, training_labels, test_data, K, tsize):
    print("Running KNN Classifier...")
    result = []
    counter = 1
    # word counts for training email
    training_WordCounts = [] 
    for training_text in training_data:
        training_WordCounts.append(get_count(training_text))
    for test_text in test_data:
        similarity = [] # List of euclidean distances
        test_WordCounts = get_count(test_text)  # word counts for test email    
        # Getting euclidean difference 
    for index in range(len(training_data)):
        euclidean_diff =\
            euclidean_difference(test_WordCounts, training_WordCounts[index])
        similarity.append([training_labels[index], euclidean_diff])
    # Sort list in ascending order based on euclidean difference
    similarity = sorted(similarity, key = lambda i:i[1])
    # Select K nearest neighbours
    selected_Kvalues = [] 
    for i in range(K):
        selected_Kvalues.append(similarity[i])
    # Predicting the class of email
    result.append(get_class(selected_Kvalues))
    return result
def main(K):
    data = load_data()
    data = preprocess_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)    
    tsize = len(test_data)
    result = knn_classifier(training_data, training_labels, test_data[:tsize], K, tsize) 
    accuracy = accuracy_score(test_labels[:tsize], result)
    print("training data size\t: " + str(len(training_data)))
    print("test data size\t\t: " + str(len(test_data)))
    print("K value\t\t\t\t: " + str(K))
    print("Samples tested\t\t: " + str(tsize))
    print("% accuracy\t\t\t: " + str(accuracy * 100))
    print("Number correct\t\t: " + str(int(accuracy * tsize)))
    print("Number wrong\t\t: " + str(int((1 - accuracy) * tsize)))
main(11)   
##########################
# import dearpygui.dearpygui as dpg
# #DearPyGUI Imports
# from dearpygui.core import *
# from dearpygui.simple import *

# #functions.py Imports
# from functions import categorize_words, pre_process, predict

# pred = []
# #button callbak function
# #runs each time when the "Check" button is clicked
# def check_spam(pred):
#     with window("Simple SMS Spam Filter"):
#         if pred == []:
#             #runs only once - the the button is first clicked
#             #and pred[-1] widget doesn't exist
#             add_spacing(count=12)
#             add_separator()
#             add_spacing(count=12)
#         else:
#             #hide prediction widget
#             hide_item(pred[-1])
#         #collect input, pre-process and get prediction
#         input_value = get_value('Input')
#         input_value = pre_process(input_value)
#         pred_text, text_colour = predict(input_value)
#         #store prediction inside the pred list
#         pred.append(pred_text)
#         #display prediction to user
#         add_text(pred[-1], color=text_colour)

# #window object settings
# set_main_window_size(540, 720)
# set_global_font_scale(1.25)
# set_theme("Gold")
# set_style_window_padding(30,30)

# with window("Simple SMS Spam Filter", width=520, height=677):
#     print("GUI is running...")
#     set_window_pos("Simple SMS Spam Filter", 0, 0)

#     #image logo
#     add_drawing("logo", width=520, height=290) #create some space for the image

#     add_separator()
#     add_spacing(count=12)
#     #text instructions
#     add_text("Please enter an SMS message of your choice to check if it's spam or not",
#     color=[232,163,33])
#     add_spacing(count=12)
#     #collect input
#     add_input_text("Input", width=415, default_value="type message here!")
#     add_spacing(count=12)
#     #action button
#     add_button("Check", callback=lambda x,y:check_spam(pred))
 
# draw_image("logo", "logo_spamFilter.png", [0,0], [458,192])

# start_dearpygui()
# print("Bye Bye, GUI")