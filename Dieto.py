# Name: Ami Patel and Jay Patel
# Class: CS 4395
# Date: December 1, 2019
# Project: ChatBot Project

# necessary imports:
import nltk
import re
from os import path
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reading the existing knowledge base
f = open('kb.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()
raw = ' '.join(raw.split())

# Extracting the sentence and word tokens from the raw data
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# The greeting messages for the bot
inputs = ("hello", "hi", "greetings", "hey")

# Function to respond to the greetings from the user
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in inputs:
            return 'Hey, let\'s start with your name. Please introduce yourself'
    return None

# The list of simple comments that the user can make
simpleChat = [['PRP', 'VBP', 'DT', 'JJ', 'NN'],
              ['PRP', 'VBP', 'DT', 'NN'],
              ['PRP', 'VBP', 'JJ'],
              ['PRP', 'VBP', 'JJ', 'NN']]


# Helper functions to find different parts of speech from the user's response using POS Tags

# Finding pronouns
def find_prp(s):
    response = None

    for w, POS in s.pos_tags:
        if w == 'you':
            response = 'I'
        if w == 'i':
            response = 'You'

    return response

# Finding nouns
def find_nn(s):
    nn = None

    for w, POS in s.pos_tags:
        if 'NN' in POS:
            nn = w

    return nn

# Finding verbs
def find_vp(s):
    vb = None

    for w, POS in s.pos_tags:
        if 'VB' in POS:
            if w == 'am':
                vb = 'are'
            elif w == 'are':
                vb = 'am'
            else:
                vb = w

    return vb

# Finding adjectives
def find_adj(s):
    adj = None

    for w, POS in s.pos_tags:
        if 'JJ' in POS:
            adj = w

    return adj

# Finding determinants
def find_dt(s):
    dt = None

    for w, POS in s.pos_tags:
        if 'DT' in POS:
            dt = w

    return dt

# Function to find all the parts of speech in a given sentence
def find_parts_of_speech(parsed):
    for s in parsed.sentences:
        prp_response = find_prp(s)
        nn_response = find_nn(s)
        vb_response = find_vp(s)
        adj_response = find_adj(s)
        dt_response = find_dt(s)
    return prp_response, nn_response, vb_response, adj_response, dt_response


# Function to return the response by the bot
def response2(user):
    sent_tokens.append(user)

    # Preprocessing the raw user_response
    userl = user.lower()
    punctuations = '''!()-[]{};:"\,<>/@#$%^&*_~'''
    text = userl
    for x in userl:
        if x in punctuations:
            text = userl.replace(x, " ")
    content = re.sub(r'[^a-z0-9 .?]+', ' ', text)

    # Using TextBlob to get the POS Tags
    parsed = TextBlob(content)
    prp, nn, vb, adj, dt = find_parts_of_speech(parsed)
    pos = []
    for w, POS in parsed.pos_tags:
        pos.append(POS)
    res = ''

    # if it is a question, form the answer from the knowledge base
    if '?' in user:
        return response(user)

    # if it is a simple comment, respond according to the different parts of speech gathered from the user's comment
    elif pos in simpleChat:
        res = 'Sure, '
        if parsed.noun_phrases is not []:
            if nn is not None:
                if dt is None and adj is not None:
                    res += prp + ' ' + vb + ' ' + adj + ' ' + nn + '!'
                elif dt is None and adj is None:
                    res += prp + ' ' + vb + ' ' + nn + '!'
                elif dt is not None and adj is None:
                    res += prp + ' ' + vb + ' ' + dt + ' ' + nn + '!'
                else:
                    res += prp + ' ' + vb + ' ' + dt + ' ' + adj + ' ' + nn + '!'
            else:
                if dt is None and adj is not None:
                    res += prp + ' ' + vb + ' ' + adj + '!'
                elif dt is None and adj is None:
                    res += prp + ' ' + vb + '!'
                elif dt is not None and adj is None:
                    res += prp + ' ' + vb + ' ' + dt + '!'
                else:
                    res += prp + ' ' + vb + ' ' + dt + ' ' + adj + '!'
        elif adj is not None:
            res += prp + ' ' + vb + ' ' + adj + '!'
        else:
            if dt is not None and nn is not None:
                res += prp + ' ' + vb + ' ' + dt + ' ' + nn + '!'
        # Since it was a simple comment, no need to keep it in the sentences list
        sent_tokens.remove(user)
        return res
    else:
        # If it is a simple statement requiring more explanation, then look in the knowledge base for the response
        return response(user)


def response(user_response):

    # For statements that need more explanation and questions related to diet,
    bot_response = ''

    # pre-process the user's response
    user_response = user_response.lower()
    punctuations = '''!()-[]{};:"\,<>/@#$%^&*_~'''
    for x in user_response:
        if x in punctuations:
            user_response = user_response.replace(x, " ")
    content = re.sub(r'[^a-z0-9 .?]+', ' ', user_response)
    use = ''

    if '?' in content:
        # Check for any general bot related questions, if so return the response immediately
        if (content == 'how are you?' or content == 'what is your name?'):
            if content == 'how are you?':
                bot_response = 'I am fine!'
            else:
                bot_response = 'My name is DIETO. I will answer your queries about diets.'
            sent_tokens.remove(user_response)
            return bot_response

    # Create the vector space model with all the sentences
    TfidfVec = TfidfVectorizer()
    tfidf = TfidfVec.fit_transform(sent_tokens)
    similar_vector_vals = cosine_similarity(tfidf[-1], tfidf)

    # the second last sentence will have the highest cosine similarity
    sentence_number = similar_vector_vals.argsort()[0][-2]

    # Flatten to check if the cosine similarity is 0 or not
    flat = similar_vector_vals.flatten()
    flat.sort()
    actual_tfidf = flat[-2]

    # If nothing relevant is found,
    if (actual_tfidf == 0):
        bot_response = bot_response + "I am sorry! I don't understand you"
        sent_tokens.remove(user_response)
        return bot_response
    # else if something relevant is found, respond to the user
    else:
        bot_response = bot_response + sent_tokens[sentence_number]
        sent_tokens.remove(user_response)
        return bot_response

# The bot starts:
Bye = False
print("DIETO: My name is DIETO. I will answer your queries about diets. If you want to exit, type Bye!")
file = 'users.txt'
user_data = []

while (Bye is not True):
    # keep the conversation going
    print("YOU: ", end='')
    user_response = input()
    user_response = user_response.lower()
    user_data.append(user_response)
    if (user_response != 'bye'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            respond = "You are welcome."
            print("DIETO:" + respond)
            user_data.append(respond)
        else:
            if (greeting(user_response) != None):

                # if the user greets, take user's name and start storing the information users.txt
                respond = greeting(user_response)
                print("DIETO: " + respond)
                user_data.append(respond)
                print("YOU: ", end='')
                user_response = input()
                user_data.append(user_response)

                if (user_response != 'bye'):
                    p = TextBlob(user_response)
                    names = []

                    # The data gets written to the users.txt
                    if path.exists(file):
                        f = open(file, 'r')
                        for line in f:
                            if ':' in line:
                                data = line.split(':')
                                name = data[0]
                                names.append(name)
                        f.close()

                    new_name = p.noun_phrases
                    if new_name[0] not in names:
                        # If the name is new, greet them in a friendly manner and record the data about their likes and dislikes
                        new = True

                        respond = "Nice to meet you, " + new_name[0] + "! Do you want to know more about healthy diet plans?"
                        print("DIETO: " + respond)
                        user_data.append(respond)

                        print("YOU: ", end='')
                        user_response = input()
                        user_response = user_response.lower()

                        # Starts the conversation depending on whether the user likes to talk about diets or not
                        if 'yes' in user_response:
                            user_data.append('LIKE-' + user_response)
                            respond = "Perfect! Let's get started. You can ask me any questions you might have!"
                            print("DIETO: " + respond)
                            user_data.append(respond)

                            print("YOU: ", end='')
                            user_response = input()
                            user_data.append(user_response)
                            user_response.lower()

                            if (user_response == 'bye'):
                                Bye = True
                                respond = "Bye! take care.."
                                print("DIETO:" + respond)
                                user_data.append(respond)

                            elif 'okay' in user_response or 'yeah' in user_response:
                                respond = "Glad to know you are interested in talking about healthy foods! Shoot me your questions.."
                                print("DIETO: " + respond)
                                user_data.append(respond)
                            else:
                                print("DIETO: ", end="")
                                respond = response2(user_response)
                                print(respond)
                                user_data.append(respond)

                        elif 'no' in user_response:
                            # if the user doesn't like talking about diets,

                            user_data.append('DISLIKE-' + user_response)
                            respond = "I will try my best to make it more interesting for you! Shoot me your questions.."
                            print("DIETO: " + respond)
                            user_data.append(respond)

                    else:
                        # if the user has already visited before,

                        new = False
                        dislike = False
                        flag = False
                        f = open(file, 'r')

                        # read from the file if the user likes talking about diets or not
                        for line in f:
                            if flag is True:
                                if ':' not in line:
                                    if 'DISLIKE-' in line:
                                        dislike = True
                                        break
                                else:
                                    flag = False
                            else:
                                if ':' in line:
                                    p = line.split(':')
                                    name = p[0]
                                    if name == new_name[0]:
                                        flag = True
                        f.close()
                        respond = "Welcome back, " + new_name[0] + "!"

                        # respond based on user's likes,
                        if dislike is True:
                            respond += " I will try my best to make it more interesting for you! Shoot me your questions.."
                        else:
                            respond += " Glad to know you are interested in talking about healthy foods! Shoot me your questions.."
                        print("DIETO: " + respond)

                        # update the user data in users.txt
                        if dislike is True:
                            user_data.append('DISLIKE-' + respond)
                        else:
                            user_data.append('LIKE-' + respond)
                else:
                    Bye = True
                    respond = "Bye! take care.."
                    print("DIETO:" + respond)
                    user_data.append(respond)
            else:
                if 'sure' in user_response or 'yeah' in user_response or 'okay' in user_response:
                    respond = "Please feel free to ask me any further questions you might have! If you want to exit, type Bye!"
                    print("DIETO: " + respond)
                    user_data.append(respond)
                    continue

                # if the user's response is not a greeting message,
                print("DIETO: ", end="")

                # look in the knowledge base for the response
                respond = response2(user_response)
                print(respond)
                user_data.append(respond)

    else:
        Bye = True
        respond = "Bye! take care.."
        print("DIETO:" + respond)
        user_data.append(respond)

if new is True:
    # if it was a new user, add the user data to users.txt
    with open(file, 'a+') as f:
        line = new_name[0] + ':' + '\n'
        for data in user_data:
            line += data + '\n'
        f.write(line)
else:
    # if the user visited before, update the users.txt file with the new data
    f = open(file, 'r')
    lines = f.read().split('\n')
    f.close()
    f = open(file, 'w')
    flag = False
    for line in lines:
        if flag is True:
            if ':' not in line:
                continue
            else:
                f.write(line + '\n')
                flag = False
        else:
            if ':' in line:
                p = line.split(':')
                name = p[0]
                if name == new_name[0]:
                    d = new_name[0] + ':' + '\n'
                    for data in user_data:
                        d += data + '\n'
                    f.write(d)
                    flag = True
                else:
                    flag = False
                    d = name + ':\n'
                    f.write(d)
            else:
                f.write(line + '\n')
    f.close()
