import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from os import listdir 
from os.path import isfile, join 
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser(description="Generate term-document matrix.")
parser.add_argument("-T", "--tfidf", action="store_true", help="Apply tf-idf to the matrix.")
parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                    default=None,
                    help="Use TruncatedSVD to truncate to N dimensions")
parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                    type=int, default=None,
                    help="Use the top M dims from the raw counts before further processing")
parser.add_argument("foldername", type=str,
                    help="The base folder name containing the two topic subfolders.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the matrix data.")

# intersection
def inters(xs,ys):
    return [x for x in xs for y in ys if x == y]

# # intersection
# # enumerate to get this out
# def intersnames(xs,ys):
#     return [x for x in xs for y in ys if x == y]

# delete a single element from a list if it exists
def minus1(x,ys):
    z = []
    for y in ys:
        if y != x:
            z.append(y)
    return z

# ys - Intersection(xs,ys)
def minusxs(xs,ys):
    for x in xs:
        ys = minus1(x,ys)
    return ys

def remove_dupl(first,second):
    inter_firstsecond = inters(first,second)
    fm_inter_firstsecond = minusxs(inter_firstsecond,first)
    newmat = fm_inter_firstsecond + second
    return newmat
    # return inter_firstsecond

def read_file(path,fn,name):
    with open(path + name + fn ) as train_file:
        rd = train_file.read()
    return rd

def read_files(path,xs,name):
    l = []
    for x in xs:
        l.append(read_file(path,x,name))
    return l

# matrix_to_list :: Matrix class 'numpy.int64' -> [[Int]]
# toarray gives the actual numpy array
# to list returns a naive python list
def matrix_to_list(matrix):
    matrix = matrix.toarray()
    return matrix.tolist()

# transpose
def tr(xs):
    return [[x[i] for x in xs] for i in range(len(xs[0]))]

# this basically reads all the files in the immediate subdirectories
def ff(path):
    file_dirs = [f for f in listdir(path)] 
    path1 = path + file_dirs[0]
    path2 = path + file_dirs[1]
    f1 = listdir(path1)
    f1names = [file_dirs[0] + '/' + f for f in f1]
    f2 = listdir(path2)
    f2names = [file_dirs[1] + '/' + f for f in f2]
    a = read_files(path,f1,file_dirs[0] + '/')
    b = read_files(path,f2,file_dirs[1] + '/')
    # return a + b, len(f1), f1, f2 
    return a + b, len(f1), f1names, f2names

# makeDict :: {Str : Int} -> [Str] -> {Str : Int}
def makeDict(d,xs):
    for key in xs:
        if key in d:
            d[key] += 1
        else:
            d[key] = 1
    return d

# make big dict
# makeBigDictionary :: [[Str]] -> {Str : Int}
def makeBigDictionary(xs):
    d = {}
    for doc in xs:
        d = makeDict(d,doc)
    return d

# vectorcounts
# :: Dict -> [Str] -> [Int]
def my_vectorizer(d,xs):
    ld = len(d)
    zeros = [0 for x in range(ld)]
    newD = dict(zip(sorted(d),zeros))
    for key in xs:
        if key in d:
            newD[key] += 1
    v = [x[1] for x in newD.items()]
    feature_names = [x[0] for x in newD.items()]
    # return v
    return v, feature_names

def filter_dict(lbd,n):
    itms = lbd.items()
    sorted_itms = sorted(itms, key=lambda it: it[1],reverse =True)
    topn_itms = sorted_itms[:n]
    return dict(topn_itms)

# tokenize :: [Str] -> [[Str]]
def tokenize(xs):
    tokens = [[x.lower() for x in token_pattern.findall(z)] for z in all_docs]
    return tokens

# write_out :: Str -> [Str] -> [Str] -> [[Int]] -> IO ()
def write_out(file_name,rownames,column_names,data):
    data = tr(data)
    df = pd.DataFrame(data=data,index=rownames,columns=column_names)
    df.to_csv(file_name)

# write_out :: Str -> [Str] -> [Str] -> [[Int]] -> IO ()
def write_out2(file_name,rownames,column_names,data):
    data = tr(matrix_to_list(data))
    df = pd.DataFrame(data=data,index=rownames,columns=column_names)
    df.to_csv(file_name)

# write_out :: Str -> [Str] -> [Str] -> [[Int]] -> IO ()
def write_out3(file_name,rownames,data):
    df = pd.DataFrame(data=data,index=rownames)
    df.to_csv(file_name)

# parameters:
args = parser.parse_args()
n_words = args.basedims
semantic_dimension = args.svddims
path0 = args.foldername
tf = args.tfidf
path2 = args.outputfile

all_docs_and_length1 = ff(path0)
all_docs = all_docs_and_length1[0]
length = all_docs_and_length1[1]
doc_names1 = all_docs_and_length1[2]
doc_names2 = all_docs_and_length1[3]
doc_names = doc_names1 + doc_names2

token_pattern = re.compile("(?u)\\b\\w\\w+\\b")
# token_pattern = re.compile("(?u)\\b[a-zA-Z][a-zA-Z]+\\b")

tokens = tokenize(all_docs)
lbd = makeBigDictionary(tokens)
 
print("Loading data from directory {}.".format(args.foldername))
if not args.basedims:
    testlist = [my_vectorizer(lbd,x)[0] for x in tokens]
    names = my_vectorizer(lbd,tokens[0])[1]
    I = pd.Index(names, name="rows")
    C = pd.Index(doc_names, name="columns")
    if args.tfidf:
        print("Applying tf-idf to raw counts.")
        tf_transformer = TfidfTransformer()
        testlist = tf_transformer.fit_transform(testlist)
        if args.svddims:
            print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
            svd = TruncatedSVD(n_components=semantic_dimension)
            testlist = svd.fit_transform(testlist)
            write_out3(path2,C,testlist)
        else: 
            write_out2(path2,I,C,testlist)
    else:
        if args.svddims:
            print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
            svd = TruncatedSVD(n_components=semantic_dimension)
            testlist = svd.fit_transform(testlist)
            write_out3(path2,C,testlist)
        else:
            write_out(path2,I,C,testlist)
    print("Using full vocabulary.")
else:
    print("Using only top {} terms by raw count.".format(args.basedims))
    flbd10 = filter_dict(lbd,n_words)
    testlist2 = [my_vectorizer(flbd10,x)[0] for x in tokens]
    names = my_vectorizer(flbd10,tokens[0])[1]
    I = pd.Index(names, name="rows")
    C = pd.Index(doc_names, name="columns")
    if args.tfidf:
        print("Applying tf-idf to raw counts.")
        tf_transformer1 = TfidfTransformer()
        testlist2 = tf_transformer1.fit_transform(testlist2)
        if args.svddims:
            print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
            svd = TruncatedSVD(n_components=semantic_dimension)
            testlist2 = svd.fit_transform(testlist2)
            write_out3(path2,C,testlist2)
        else:
            write_out2(path2,I,C,testlist2)
    else:
        if args.svddims:
            print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
            svd = TruncatedSVD(n_components=semantic_dimension)
            testlist2 = svd.fit_transform(testlist2)
            write_out3(path2,C,testlist2)
        else:
            write_out(path2,I,C,testlist2)
print("Writing matrix to {}.".format(args.outputfile))
