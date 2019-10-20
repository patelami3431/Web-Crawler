# Name: Ami Patel and Jay Patel
# Class: CS 4395
# Date: 20 October, 2019
# Project: Web-Crawler Project

# necessary imports:
from urllib import request
from bs4 import BeautifulSoup
import re
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from urllib.error import HTTPError
from nltk import word_tokenize
import math


# function to create soup for any given url
def prepare_soup(url):
    try:
        p = request.urlopen(url)
        if (p.getcode() == 403):  # check if the webpage is not forbidden
            return False
        data = p.read()
        soup = BeautifulSoup(data, 'html.parser')
        return soup
    except HTTPError as e:  # check if the webpage exists, in case of failure
        return False


# helper method to remove the unwanted elements from the text from webpage
def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True


# function to find all the necessary links from the html script available through the soup
def find_links(soup):
    s = []
    for link in soup.find_all('a'):
        s.append(link.get('href'))
    return s


# function to extract all the text from the webpage
def all_text(soup):
    # kill all script and style elements
    for script in soup(['style', 'script', '[document]', 'head', 'title']):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase for line in lines for phrase in line.split(" "))
    # drop blank lines
    text = ' '.join(chunk for chunk in chunks if chunk)
    return text


# function to tokenize the sentences and create the files with those sentences
def sent_tokenizer(filename):
    file = open(filename, "r", encoding="utf-8")
    content = file.read()
    content = content.lower()  # coverting everything to lower case
    content = ''.join([i for i in content if not i.isdigit()])

    # removing the punctuation
    punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''
    for x in content:
        if x in punctuations:
            content = content.replace(x, " ")

            # removing special symbols and digits
    content = re.sub(r'[^a-zA-Z .]+', ' ', content)

    sents = sent_tokenize(content)

    # create the files:
    with open('tokenized_' + filename, 'w+') as f:
        for s in sents:
            if "retrieved" in s or "register" in s or "reply" in s:
                continue
            else:
                f.write(s + '\n')
    f.close()


# function to create the tf_idf dictionary of unique terms across all the documents
def create_tf_idf_dict(text, num_docs):
    tf_dict = {}
    tokens = word_tokenize(text)

    # remove the stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]

    # lemmatize the tokens
    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(t) for t in tokens]

    for t in tokens:
        if t in tf_dict:
            tf_dict[t] += 1
        else:
            tf_dict[t] = 1

    for t in tf_dict.keys():
        tf_dict[t] = tf_dict[t] / len(tokens)

    vocab = set(tokens)
    idf_dict = {}
    for term in vocab:
        temp = ['x' for voc in vocab if term in voc]
        idf_dict[term] = math.log((1 + num_docs) / (1 + len(temp)))

    tf_idf = {}
    for t in tf_dict.keys():
        tf_idf[t] = tf_dict[t] * idf_dict[t]

    return tf_idf


# function to remove the excessively long sentences from the knowledge base
def Know_base(filename):
    file = open(filename, "r", encoding="utf-8")
    content = file.read()
    content = content.lower()
    content = ''.join([i for i in content if not i.isdigit()])
    punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''
    for x in content:
        if x in punctuations:
            content = content.replace(x, " ")
    content = re.sub(r'(?<=[.])(?=[^\s])', r' ', content)
    content = re.sub(r'[^a-zA-Z .]+', ' ', content)
    sents = sent_tokenize(content)
    for s in sents:
        if len(s) > 200:
            sents.remove(s)
    return sents


# WEB CRAWLER METHOD:
def web_crawler():
    soup = prepare_soup("https://en.wikipedia.org/wiki/List_of_diets")
    s = find_links(soup)
    counter = 0
    final = []
    crawl = True
    i = 0

    # cral untli we get at least 15 relevant urls:
    while (crawl):
        for link in s:
            if link and ('diet' in link or 'Diet' in link) and link.startswith('http'):
                final.append(link)
                counter += 1
        if (counter > 15):
            crawl = False
            break
        else:
            soup1 = prepare_soup(final[i])
            while soup1 is False:
                i += 1
                soup1 = prepare_soup(final[i])
                s = find_links(soup1)
                continue
            i += 1
            s = find_links(soup1)

            # Store the text from the urls to different files
    file = "file_"
    j = 0
    files = []
    for url in final:
        soup2 = prepare_soup(url)
        if soup2 is False:
            continue
        text = all_text(soup2)
        filename = file + str(j) + ".txt"
        if text:
            with open(filename, "w+", encoding="utf-8") as f:
                f.write(text)
            files.append(filename)
            f.close()
            j += 1
    print("The total number of files:", j)

    # Clean up the text and extract sentences:
    vocab = []
    terms = []
    text = ''

    for f in files:
        sent_tokenizer(f)
        for s in Know_base(f):
            if 'jpg' not in s:
                terms.append(s)
    for f in files:
        with open("tokenized_" + f, "r") as file:
            text += file.read() + " "
        file.close()

    text = text.lower()
    text = re.sub(r'[.?!,:;()\-]', ' ', text)

    # extract at least 10 important terms from the pages using tf-idf
    tf_idf_dict = create_tf_idf_dict(text, len(files))

    l = 0

    print("The first 35 important words according to tf_idf dictionary are: ")
    for k in sorted(tf_idf_dict, key=lambda k: tf_idf_dict[k], reverse=True):
        if l < 35:
            print(k, "->", tf_idf_dict[k])
            l += 1

    # The top 10 important terms:
    Term_list = ['calorie', 'weight', 'food', 'eating', 'loss', 'fasting', 'disease', 'body', 'protein', 'meal']

    # Build a searchable knowledge base:
    Term_dict = {}
    a = []
    for t in Term_list:
        for s in terms:
            if t in s:
                a.append((t, s))
    for t, s in a:
        Term_dict.setdefault(t, []).append(s)

    file = "kb.txt"
    with open(file, 'w') as f:
        for token in Term_dict.keys():
            line = token + ':' + '\n'
            i = 1
            for sent in Term_dict[token]:
                line += str(i) + '. \'' + sent + '\',' + '\n'
                i += 1
            f.write(line + '\n')


web_crawler()