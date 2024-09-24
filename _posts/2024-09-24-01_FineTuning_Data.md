# FineTuning
FineTuning을 해볼 겁니다.  
코드는 테디노트에서 따왔어요. 제가 보려고 만든 블로그이니 저게 원하는 주석을 달아가면서 진행해보려고 합니다.  
<br>
자 시작해 보겠습니다. 먼저 .env 변수들을 불러보겠습니다.  


```python

from dotenv import load_dotenv

load_dotenv()
```




    True



## LangSmith


```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("My-Book-02-FineTuning")
```

    LangSmith 추적을 시작합니다.
    [프로젝트명]
    My-Book-02-FineTuning




## QA Pair용 PDF 로드
QA Pair를 생성할 PDF를 로드합니다.  
테디님은 SPRI AI Brief 파일을 좋아하시길래 동일한 파일을 준비해 보았습니다.  
URL : https://spri.kr/lib/fileman/Uploads/post_images/2023_12/1208.jpg  

unstructured 라이브러리는 다양한 형식의 비정형 데이터를 처리할 수 있어요.  
Text, PDF, Word, HTML, Image 등을 예로 들 수 있습니다.  
<br>
partition_pdf를 실행할 때 nltk를 사용하니 실행전에 우선 다운을 다 받아 놓습니다.  
NLTK(Natural Language Toolkit)는 자연어 처리를 위한 강력한 Python 라이브러리에요.  
1. 토큰화(Tokenization).
    - 문장이나 단어 단위로 텍스트를 나누는 기능을 제공합니다.
    - 주요 모듈: word_tokenize, sent_tokenize
2. 품사 태깅(Part-of-Speech Tagging)
    - 각 단어에 대해 품사(명사, 동사 등)를 태깅합니다.
    - 주요 모듈: pos_tag
3. 어간 추출(Stemming)
    - 단어의 어근을 추출하는 기능입니다. 이를 통해 단어 형태 변화를 일반화할 수 있습니다.
    - 주요 알고리즘: PorterStemmer, LancasterStemmer
4. 표제어 추출(Lemmatization)
    - 단어의 기본 형태(표제어)를 추출합니다.
    - 주요 모듈: WordNetLemmatizer
5. 정규식 처리(Regular Expressions)
    - 정규 표현식을 활용하여 텍스트 데이터를 처리할 수 있습니다.
    - 주요 모듈: RegexpTokenizer
6. 구문 분석(Parsing)
    - 텍스트에서 구문 구조를 분석하는 기능입니다.
    - 주요 모듈: RecursiveDescentParser, ChartParser
7. 명칭 인식(Named Entity Recognition, NER)
    - 사람, 장소, 조직명 등과 같은 명명된 엔티티를 인식합니다.
    - 주요 모듈: ne_chunk
8. 텍스트 분류(Text Classification)
    - 문서를 카테고리로 분류할 수 있습니다.
    - 주요 모듈: NaiveBayesClassifier, DecisionTreeClassifier
9. 텍스트 유사도 계산(Text Similarity)
    - 두 텍스트 간의 유사도를 측정하는 기능을 제공합니다.
    - 주요 모듈: edit_distance, jaccard_distance
10. 코퍼스 데이터(Corpus Data)
    - 다양한 코퍼스(말뭉치)를 포함하고 있습니다. 예를 들어, 영어 텍스트, 뉴스, 문학 작품 등의 데이터를 이용할 수 있습니다.
    - 주요 코퍼스: gutenberg, brown, reuters
11. 워드넷(WordNet)
    - 영어 단어의 의미 관계를 탐색할 수 있는 대규모 어휘 데이터베이스입니다.
    - 주요 모듈: wordnet
12. 언어 모델(Language Models)
    - 언어 모델을 생성하고, 특정 언어에서 단어의 확률을 예측할 수 있습니다.
13. 스톱워드(Stopwords)
    - 텍스트에서 자주 사용되지만 의미가 없는 단어(예: 'the', 'is')를 쉽게 제거할 수 있습니다.
    - 주요 모듈: stopwords
14. n-그램(N-grams)
    - n-gram을 생성하여 텍스트의 연속된 n개의 단어를 분석하는 데 사용할 수 있습니다.
    - 주요 모듈: ngrams
15. 번역(Translation) 및 음성 분석(Phonetic Analysis)
    - 단어의 발음 기호를 분석하거나 번역을 수행할 수 있는 도구도 일부 포함되어 있습니다.
16. 단어 빈도 분석(Frequency Analysis)
    - 텍스트 내에서 단어 빈도를 분석하는 기능을 제공합니다.
    - 주요 모듈: FreqDist
17. 트리 구조(Tree Representation)
    - 구문 분석된 문장을 트리 구조로 표현하는 기능을 제공합니다.
18. 감정 분석(Sentiment Analysis)
    - 문장의 감정(긍정, 부정 등)을 분석할 수 있는 기능도 제공됩니다. 


```python
import nltk
nltk.download('all')
```

    [nltk_data] Downloading collection 'all'
    [nltk_data]    | 
    [nltk_data]    | Downloading package abc to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/abc.zip.
    [nltk_data]    | Downloading package alpino to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/alpino.zip.
    [nltk_data]    | Downloading package averaged_perceptron_tagger to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping taggers/averaged_perceptron_tagger.zip.
    [nltk_data]    | Downloading package averaged_perceptron_tagger_eng to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping
    [nltk_data]    |       taggers/averaged_perceptron_tagger_eng.zip.
    [nltk_data]    | Downloading package averaged_perceptron_tagger_ru to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping
    [nltk_data]    |       taggers/averaged_perceptron_tagger_ru.zip.
    [nltk_data]    | Downloading package averaged_perceptron_tagger_rus to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping
    [nltk_data]    |       taggers/averaged_perceptron_tagger_rus.zip.
    [nltk_data]    | Downloading package basque_grammars to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping grammars/basque_grammars.zip.
    [nltk_data]    | Downloading package bcp47 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package biocreative_ppi to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/biocreative_ppi.zip.
    [nltk_data]    | Downloading package bllip_wsj_no_aux to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping models/bllip_wsj_no_aux.zip.
    [nltk_data]    | Downloading package book_grammars to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping grammars/book_grammars.zip.
    [nltk_data]    | Downloading package brown to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/brown.zip.
    [nltk_data]    | Downloading package brown_tei to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/brown_tei.zip.
    [nltk_data]    | Downloading package cess_cat to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/cess_cat.zip.
    [nltk_data]    | Downloading package cess_esp to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/cess_esp.zip.
    [nltk_data]    | Downloading package chat80 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/chat80.zip.
    [nltk_data]    | Downloading package city_database to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/city_database.zip.
    [nltk_data]    | Downloading package cmudict to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/cmudict.zip.
    [nltk_data]    | Downloading package comparative_sentences to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/comparative_sentences.zip.
    [nltk_data]    | Downloading package comtrans to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package conll2000 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/conll2000.zip.
    [nltk_data]    | Downloading package conll2002 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/conll2002.zip.
    [nltk_data]    | Downloading package conll2007 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package crubadan to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/crubadan.zip.
    [nltk_data]    | Downloading package dependency_treebank to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/dependency_treebank.zip.
    [nltk_data]    | Downloading package dolch to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/dolch.zip.
    [nltk_data]    | Downloading package europarl_raw to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/europarl_raw.zip.
    [nltk_data]    | Downloading package extended_omw to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package floresta to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/floresta.zip.
    [nltk_data]    | Downloading package framenet_v15 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/framenet_v15.zip.
    [nltk_data]    | Downloading package framenet_v17 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/framenet_v17.zip.
    [nltk_data]    | Downloading package gazetteers to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/gazetteers.zip.
    [nltk_data]    | Downloading package genesis to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/genesis.zip.
    [nltk_data]    | Downloading package gutenberg to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/gutenberg.zip.
    [nltk_data]    | Downloading package ieer to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/ieer.zip.
    [nltk_data]    | Downloading package inaugural to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/inaugural.zip.
    [nltk_data]    | Downloading package indian to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/indian.zip.
    [nltk_data]    | Downloading package jeita to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package kimmo to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/kimmo.zip.
    [nltk_data]    | Downloading package knbc to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package large_grammars to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping grammars/large_grammars.zip.
    [nltk_data]    | Downloading package lin_thesaurus to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/lin_thesaurus.zip.
    [nltk_data]    | Downloading package mac_morpho to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/mac_morpho.zip.
    [nltk_data]    | Downloading package machado to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package masc_tagged to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package maxent_ne_chunker to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping chunkers/maxent_ne_chunker.zip.
    [nltk_data]    | Downloading package maxent_ne_chunker_tab to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping chunkers/maxent_ne_chunker_tab.zip.
    [nltk_data]    | Downloading package maxent_treebank_pos_tagger to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping taggers/maxent_treebank_pos_tagger.zip.
    [nltk_data]    | Downloading package maxent_treebank_pos_tagger_tab to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping
    [nltk_data]    |       taggers/maxent_treebank_pos_tagger_tab.zip.
    [nltk_data]    | Downloading package moses_sample to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping models/moses_sample.zip.
    [nltk_data]    | Downloading package movie_reviews to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/movie_reviews.zip.
    [nltk_data]    | Downloading package mte_teip5 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/mte_teip5.zip.
    [nltk_data]    | Downloading package mwa_ppdb to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping misc/mwa_ppdb.zip.
    [nltk_data]    | Downloading package names to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/names.zip.
    [nltk_data]    | Downloading package nombank.1.0 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package nonbreaking_prefixes to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/nonbreaking_prefixes.zip.
    [nltk_data]    | Downloading package nps_chat to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/nps_chat.zip.
    [nltk_data]    | Downloading package omw to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package omw-1.4 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package opinion_lexicon to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/opinion_lexicon.zip.
    [nltk_data]    | Downloading package panlex_swadesh to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package paradigms to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/paradigms.zip.
    [nltk_data]    | Downloading package pe08 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/pe08.zip.
    [nltk_data]    | Downloading package perluniprops to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping misc/perluniprops.zip.
    [nltk_data]    | Downloading package pil to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/pil.zip.
    [nltk_data]    | Downloading package pl196x to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/pl196x.zip.
    [nltk_data]    | Downloading package porter_test to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping stemmers/porter_test.zip.
    [nltk_data]    | Downloading package ppattach to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/ppattach.zip.
    [nltk_data]    | Downloading package problem_reports to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/problem_reports.zip.
    [nltk_data]    | Downloading package product_reviews_1 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/product_reviews_1.zip.
    [nltk_data]    | Downloading package product_reviews_2 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/product_reviews_2.zip.
    [nltk_data]    | Downloading package propbank to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package pros_cons to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/pros_cons.zip.
    [nltk_data]    | Downloading package ptb to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/ptb.zip.
    [nltk_data]    | Downloading package punkt to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Package punkt is already up-to-date!
    [nltk_data]    | Downloading package punkt_tab to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping tokenizers/punkt_tab.zip.
    [nltk_data]    | Downloading package qc to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/qc.zip.
    [nltk_data]    | Downloading package reuters to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package rslp to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping stemmers/rslp.zip.
    [nltk_data]    | Downloading package rte to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/rte.zip.
    [nltk_data]    | Downloading package sample_grammars to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping grammars/sample_grammars.zip.
    [nltk_data]    | Downloading package semcor to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package senseval to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/senseval.zip.
    [nltk_data]    | Downloading package sentence_polarity to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/sentence_polarity.zip.
    [nltk_data]    | Downloading package sentiwordnet to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/sentiwordnet.zip.
    [nltk_data]    | Downloading package shakespeare to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/shakespeare.zip.
    [nltk_data]    | Downloading package sinica_treebank to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/sinica_treebank.zip.
    [nltk_data]    | Downloading package smultron to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/smultron.zip.
    [nltk_data]    | Downloading package snowball_data to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package spanish_grammars to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping grammars/spanish_grammars.zip.
    [nltk_data]    | Downloading package state_union to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/state_union.zip.
    [nltk_data]    | Downloading package stopwords to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/stopwords.zip.
    [nltk_data]    | Downloading package subjectivity to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/subjectivity.zip.
    [nltk_data]    | Downloading package swadesh to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/swadesh.zip.
    [nltk_data]    | Downloading package switchboard to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/switchboard.zip.
    [nltk_data]    | Downloading package tagsets to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping help/tagsets.zip.
    [nltk_data]    | Downloading package tagsets_json to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping help/tagsets_json.zip.
    [nltk_data]    | Downloading package timit to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/timit.zip.
    [nltk_data]    | Downloading package toolbox to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/toolbox.zip.
    [nltk_data]    | Downloading package treebank to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/treebank.zip.
    [nltk_data]    | Downloading package twitter_samples to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/twitter_samples.zip.
    [nltk_data]    | Downloading package udhr to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/udhr.zip.
    [nltk_data]    | Downloading package udhr2 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/udhr2.zip.
    [nltk_data]    | Downloading package unicode_samples to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/unicode_samples.zip.
    [nltk_data]    | Downloading package universal_tagset to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping taggers/universal_tagset.zip.
    [nltk_data]    | Downloading package universal_treebanks_v20 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package vader_lexicon to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package verbnet to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/verbnet.zip.
    [nltk_data]    | Downloading package verbnet3 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/verbnet3.zip.
    [nltk_data]    | Downloading package webtext to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/webtext.zip.
    [nltk_data]    | Downloading package wmt15_eval to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping models/wmt15_eval.zip.
    [nltk_data]    | Downloading package word2vec_sample to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping models/word2vec_sample.zip.
    [nltk_data]    | Downloading package wordnet to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package wordnet2021 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package wordnet2022 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/wordnet2022.zip.
    [nltk_data]    | Downloading package wordnet31 to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    | Downloading package wordnet_ic to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/wordnet_ic.zip.
    [nltk_data]    | Downloading package words to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/words.zip.
    [nltk_data]    | Downloading package ycoe to
    [nltk_data]    |     /Users/dorumugs/nltk_data...
    [nltk_data]    |   Unzipping corpora/ycoe.zip.
    [nltk_data]    | 
    [nltk_data]  Done downloading collection all





    True




```python
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(filepath):
    """
    PDF 파일에서 이미지, 테이블, 그리고 텍스트 조각을 추출합니다.
    path: 이미지(.jpg)를 저장할 파일 경로
    fname: 파일 이름
    """
    return partition_pdf(
        filename=filepath,
        extract_images_in_pdf=False,  # PDF 내 이미지 추출 활성화
        infer_table_structure=False,  # 테이블 구조 추론 활성화
        chunking_strategy="by_title",  # 제목별로 텍스트 조각화
        max_characters=4000,  # 최대 문자 수
        new_after_n_chars=3800,  # 이 문자 수 이후에 새로운 조각 생성
        combine_text_under_n_chars=2000,  # 이 문자 수 이하의 텍스트는 결합
    )
```


```python
# PDF 파일 로드
elements = extract_pdf_elements("data/SPRI_AI_Brief_2023년12월호_F.pdf")
# 로드한 TEXT 청크 수
len(elements)
```




    13



Print 해보면 추출된 내역을 확인 할 수 있어요.  
FineTuning을 하려면, 이렇게 추출된 데이터를 Question + Answer 형태로 구현해야 합니다.  
JSON 형태로 구성해보면 아래와 같아요.  
<br>
{{  
"QUESTION": "바이든 대통령이 서명한 '안전하고 신뢰할 수 있는 AI 개발과 사용에 관한 행정명령'의 주요 목적 중 하나는 무엇입니까?",  
"ANSWER": "바이든 대통령이 서명한 행정명령의 주요 목적은 AI의 안전 마련과 보안 기준 마련을 위함입니다."  
}}  


```python
print(type(elements[2]))
# print("elemnets 0 : ", elements[0])
print("elemnets 1 : ", elements[1])
```

    <class 'unstructured.documents.elements.CompositeElement'>
    elemnets 1 :  KEY Contents
    
    n 미국 바이든 대통령이 ‘안전하고 신뢰할 수 있는 AI 개발과 사용에 관한 행정명령’에 서명하고
    
    광범위한 행정 조치를 명시
    
    n 행정명령은 △AI의 안전과 보안 기준 마련 △개인정보보호 △형평성과 시민권 향상 △소비자
    
    보호 △노동자 지원 △혁신과 경쟁 촉진 △국제협력을 골자로 함
    
    £ 바이든 대통령, AI 행정명령 통해 안전하고 신뢰할 수 있는 AI 개발과 활용 추진
    
    n 미국 바이든 대통령이 2023년 10월 30일 연방정부 차원에서 안전하고 신뢰할 수 있는 AI 개발과
    
    사용을 보장하기 위한 행정명령을 발표
    
    행정명령은 △AI의 안전과 보안 기준 마련 △개인정보보호 △형평성과 시민권 향상 △소비자 보호
    
    △노동자 지원 △혁신과 경쟁 촉진 △국제협력에 관한 내용을 포괄
    
    n (AI 안전과 보안 기준) 강력한 AI 시스템을 개발하는 기업에게 안전 테스트 결과와 시스템에 관한
    
    주요 정보를 미국 정부와 공유할 것을 요구하고, AI 시스템의 안전성과 신뢰성 확인을 위한 표준 및
    
    AI 생성 콘텐츠 표시를 위한 표준과 모범사례 확립을 추진
    
    △1026 플롭스(FLOPS, Floating Point Operation Per Second)를 초과하는 컴퓨팅 성능 또는 생물학적 서열 데이터를 주로 사용하고 1023플롭스를 초과하는 컴퓨팅 성능을 사용하는 모델 △단일 데이터센터에서 1,000Gbit/s 이상의 네트워킹으로 연결되며 AI 훈련에서 이론상 최대 1020 플롭스를 처리할 수 있는 컴퓨팅 용량을 갖춘 컴퓨팅 클러스터가 정보공유 요구대상
    
    n (형평성과 시민권 향상) 법률, 주택, 보건 분야에서 AI의 무책임한 사용으로 인한 차별과 편견 및 기타
    
    문제를 방지하는 조치를 확대
    
    형사사법 시스템에서 AI 사용 모범사례를 개발하고, 주택 임대 시 AI 알고리즘 차별을 막기 위한 명확한
    
    지침을 제공하며, 보건복지 부문에서 책임 있는 AI 배포와 사용을 위한 전략을 마련
    
    n (소비자 보호와 근로자 지원) 의료 분야에서 책임 있는 AI 사용을 촉진하고 맞춤형 개인교습 등 학교
    
    내 AI 교육 도구 관련 자원을 개발하며, AI로 인한 근로자 피해를 완화하고 이점을 극대화하는 원칙과
    
    모범사례를 마련
    
    n (혁신과 경쟁 촉진) 국가AI연구자원(National Artificial Intelligence Research Resource, NAIRR)*을
    
    통해 미국 전역의 AI 연구를 촉진하고, 중소기업과 개발자에 기술과 인프라를 지원
    
    국가 차원에서 AI 연구 인프라를 확충해 더 많은 AI 연구자에게 인프라를 지원하는 프로그램
    
    비자 기준과 인터뷰 절차의 현대화와 간소화로 AI 관련 주요 분야의 전문 지식을 갖춘 외국인들이 미국에서
    
    공부하고 취업할 수 있도록 지원
    
    ☞ 출처 : The White House, Executive Order on the Safe, Secure, and Trustworthy Development and Use of
    
    Artificial Intelligence (E.O. 14110), 2023.10.30.
    
    SPRi AI Brief | 2023-12월호
    
    G7, 히로시마 AI 프로세스를 통해 AI 기업 대상 국제 행동강령에 합의
    
    KEY Contents
    
    n G7이 첨단 AI 시스템을 개발하는 기업을 대상으로 AI 위험 식별과 완화를 위해 자발적인
    
    채택을 권고하는 AI 국제 행동강령을 마련
    
    n 행동강령은 AI 수명주기 전반에 걸친 위험 평가와 완화, 투명성과 책임성의 보장, 정보공유와
    
    이해관계자 간 협력, 보안 통제, 콘텐츠 인증과 출처 확인 등의 조치를 요구
    
    £ G7, 첨단 AI 시스템의 위험 관리를 위한 국제 행동강령 마련
    
    n 주요 7개국(G7)*은 2023년 10월 30일 ‘히로시마 AI 프로세스’를 통해 AI 기업 대상의 AI 국제
    
    행동강령(International Code of Conduct for Advanced AI Systems)에 합의
    
    G7은 2023년 5월 일본 히로시마에서 개최된 정상회의에서 생성 AI에 관한 국제규범 마련과
    
    정보공유를 위해 ‘히로시마 AI 프로세스’를 출범**
    
    기업의 자발적 채택을 위해 마련된 이번 행동강령은 기반모델과 생성 AI를 포함한 첨단 AI 시스템의
    
    위험 식별과 완화에 필요한 조치를 포함
    
    주요 7개국(G7)은 미국, 일본, 독일, 영국, 프랑스, 이탈리아, 캐나다를 의미
    
    ** 5월 정상회의에는 한국, 호주, 베트남 등을 포함한 8개국이 초청을 받았으나, AI 국제 행동강령에는 우선 G7 국가만 포함하여 채택
    
    n G7은 행동강령을 통해 아래의 조치를 제시했으며, 빠르게 발전하는 기술에 대응할 수 있도록


PDF에서 꺼내진 데이터를 Question + Answer 형태를 만들기 위해서 Prompt 만한게 없죠  
Prompt에 들어갈 {context}와 {domain} 그리고 {num_questions}를 변수 처리해서 적용합니다.  


```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """Context information is below. You are only aware of this context and nothing else.
---------------------

{context}

---------------------
Given this context, generate only questions based on the below query.
You are an Teacher/Professor in {domain}. 
Your task is to provide exactly **{num_questions}** question(s) for an upcoming quiz/examination. 
You are not to provide more or less than this number of questions. 
The question(s) should be diverse in nature across the document. 
The purpose of question(s) is to test the understanding of the students on the context information provided.
You must also provide the answer to each question. The answer should be based on the context information provided only.

Restrict the question(s) to the context information provided only.
QUESTION and ANSWER should be written in Korean. response in JSON format which contains the `question` and `answer`.
DO NOT USE List in JSON format.
ANSWER should be a complete sentence.

#Format:
```json
{{
    "QUESTION": "바이든 대통령이 서명한 '안전하고 신뢰할 수 있는 AI 개발과 사용에 관한 행정명령'의 주요 목적 중 하나는 무엇입니까?",
    "ANSWER": "바이든 대통령이 서명한 행정명령의 주요 목적은 AI의 안전 마련과 보안 기준 마련을 위함입니다."
}},
{{
    "QUESTION": "메타의 라마2가 오픈소스 모델 중에서 어떤 유형의 작업에서 가장 우수한 성능을 발휘했습니까?",
    "ANSWER": "메타의 라마2는 RAG 없는 질문과 답변 및 긴 형식의 텍스트 생성에서 오픈소스 모델 중 가장 우수한 성능을 발휘했습니다."    
}},
{{
    "QUESTION": "IDC 예측에 따르면 2027년까지 생성 AI 플랫폼과 애플리케이션 시장의 매출은 얼마로 전망되나요?",
    "ANSWER": "IDC 예측에 따르면 2027년까지 생성 AI 플랫폼과 애플리케이션 시장의 매출은 283억 달러로 전망됩니다."    
}}
```
"""
)
```

prompt는 만들었지만 실제 더 잘 동작하게 하려면 json 형태로 뽑아주는게 좋아요.  
아래 parser를 통해서 깔끔하게 처리하면 잡스러운 것들이 안들어가요.  
<br>
깔끔하게 진행하기 위해 response.content.strip()로 불필요한 양쪽의 공백을 제거합니다.  
<br>
.removeprefix("\`\`\`json\n")로 문자열 앞부분에 있는 "json\n"를 제거합니다.  
참고로 JSON 데이터는 종종 코드 블록으로 감싸져 있는데, 이때 앞에 "\` json\n\`"이 붙을 수 있습니다.   
<br>
.removesuffix("\n\`\`\`")로 마찬가지로 문자열의 끝에 붙어있는 "\n\`\`\`"을 제거합니다.   
코드 블록이 끝날 때 " \`\`\` `"와 같은 포맷이 붙는 경우가 있어서, 그 부분을 제거하는 작업입니다.   



```python
import json
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def custom_json_parser(response):
    json_string = response.content.strip().removeprefix("```json\n").removesuffix("\n```").strip()
    json_string = f'[{json_string}]'
    return json.loads(json_string)

chain = (
    prompt
    | ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    | custom_json_parser
)  # 체인을 구성합니다.

qa_pair = []
# 0에는 index와 제목이 들어있어서 1부터 진행합니다.
for element in elements[1:]:
    if element.text:
        qa_pair.extend(
            chain.invoke(
                {"context": element.text, "domain": "AI", "num_questions": "3"}
            )
        )
```

    ```json
    {
        "QUESTION": "바이든 대통령이 2023년 10월 30일 발표한 행정명령의 주요 내용 중 하나는 무엇입니까?",
        "ANSWER": "바이든 대통령이 발표한 행정명령의 주요 내용 중 하나는 AI의 안전과 보안 기준 마련입니다."
    },
    {
        "QUESTION": "G7이 2023년 10월 30일 합의한 AI 국제 행동강령의 주요 목적은 무엇입니까?",
        "ANSWER": "G7이 합의한 AI 국제 행동강령의 주요 목적은 AI 위험 식별과 완화를 위한 조치를 마련하는 것입니다."
    },
    {
        "QUESTION": "행정명령에 따르면 AI 시스템의 안전성과 신뢰성을 확인하기 위해 기업이 미국 정부와 공유해야 하는 정보는 무엇입니까?",
        "ANSWER": "행정명령에 따르면 AI 시스템의 안전성과 신뢰성을 확인하기 위해 기업은 안전 테스트 결과와 시스템에 관한 주요 정보를 미국 정부와 공유해야 합니다."
    }
    ``````json
    {
        "QUESTION": "블레츨리 선언에서 AI 안전 보장을 위해 강조된 이해관계자들은 누구입니까?",
        "ANSWER": "블레츨리 선언에서 AI 안전 보장을 위해 강조된 이해관계자들은 국가, 국제기구, 기업, 시민사회, 학계입니다."
    },
    {
        "QUESTION": "영국 AI 안전 연구소가 주도할 첨단 AI 모델의 안전 테스트는 어떤 잠재적 유해 기능에 대한 시험을 포함합니까?",
        "ANSWER": "영국 AI 안전 연구소가 주도할 첨단 AI 모델의 안전 테스트는 국가 안보와 안전, 사회적 피해를 포함한 여러 잠재적 유해 기능에 대한 시험을 포함합니다."
    },
    {
        "QUESTION": "G7 히로시마 프로세스 국제 행동 강령에 따르면, AI 수명주기 전반에 걸쳐 구현해야 하는 보안 통제에는 어떤 것들이 포함됩니까?",
        "ANSWER": "G7 히로시마 프로세스 국제 행동 강령에 따르면, AI 수명주기 전반에 걸쳐 구현해야 하는 보안 통제에는 물리보안, 사이버보안, 내부자 위협 보안이 포함됩니다."
    }
    ``````json
    {
        "QUESTION": "미국 캘리포니아 북부지방법원이 예술가들이 제기한 저작권 침해 소송을 기각한 주요 이유는 무엇입니까?",
        "ANSWER": "미국 캘리포니아 북부지방법원이 예술가들이 제기한 저작권 침해 소송을 기각한 주요 이유는 고소장에 제시된 상당수 작품이 저작권청에 등록되지 않았으며, AI로 생성된 이미지와 특정 작품 간 유사성을 입증하기 어렵다는 점입니다."
    },
    {
        "QUESTION": "미국 연방거래위원회(FTC)가 저작권청에 제출한 의견서에서 생성 AI로 인한 우려 사항 중 하나는 무엇입니까?",
        "ANSWER": "미국 연방거래위원회(FTC)가 저작권청에 제출한 의견서에서 생성 AI로 인한 우려 사항 중 하나는 일부 빅테크가 막대한 재원을 활용해 시장 지배력을 더욱 강화할 수 있다는 점입니다."
    },
    {
        "QUESTION": "윌리엄 오릭 판사가 예술가들에게 소송을 다시 제기할 것을 요청한 조건은 무엇입니까?",
        "ANSWER": "윌리엄 오릭 판사가 예술가들에게 소송을 다시 제기할 것을 요청한 조건은 고소장을 수정하고 저작권이 침해된 특정 이미지를 중심으로 소송 범위를 줄이는 것입니다."
    }
    ``````json
    {
        "QUESTION": "FTC가 아마존의 AI 비서 '알렉사'와 스마트홈 보안 기기 '링'에 대해 부과한 과징금은 얼마입니까?",
        "ANSWER": "FTC는 '알렉사'와 '링'에 대해 3,080만 달러(약 420억 원)의 과징금을 부과했습니다."
    },
    {
        "QUESTION": "유럽의회, EU 집행위원회, EU 이사회가 진행 중인 AI 법 최종협상에서 프랑스, 이탈리아, 독일이 제안한 기반모델 규제 방식은 무엇입니까?",
        "ANSWER": "프랑스, 이탈리아, 독일은 '의무적 자율규제(Mandatory Self-regulation)' 방식의 기반모델 규제를 제안했습니다."
    },
    {
        "QUESTION": "FTC가 AI 관련 불법 행위에 대처하기 위해 활용하는 법적 권한의 예는 무엇입니까?",
        "ANSWER": "FTC는 아마존 AI 비서 '알렉사'와 스마트홈 보안 기기 '링'이 소비자의 사적 정보를 알고리즘 훈련에 사용하여 프라이버시를 침해한 혐의를 조사하는 등의 법적 권한을 활용해 AI 관련 불법 행위에 대처하고 있습니다."
    }
    ``````json
    {
        "QUESTION": "프런티어 모델 포럼이 AI 안전 연구를 위해 조성한 기금의 규모는 얼마입니까?",
        "ANSWER": "프런티어 모델 포럼이 AI 안전 연구를 위해 조성한 기금의 규모는 1,000만 달러입니다."
    },
    {
        "QUESTION": "프런티어 모델 포럼이 AI 레드팀 활동을 지원하기 위해 중점적으로 개발하려는 것은 무엇입니까?",
        "ANSWER": "프런티어 모델 포럼이 AI 레드팀 활동을 지원하기 위해 중점적으로 개발하려는 것은 모델 평가 기법입니다."
    },
    {
        "QUESTION": "코히어가 12개 기관과 함께 공개한 '데이터 출처 탐색기' 플랫폼의 주요 목적은 무엇입니까?",
        "ANSWER": "코히어가 12개 기관과 함께 공개한 '데이터 출처 탐색기' 플랫폼의 주요 목적은 데이터 투명성을 향상시키는 것입니다."
    }
    ``````json
    {
        "QUESTION": "연구진이 데이터 출처 탐색기를 통해 해결하고자 하는 주요 문제는 무엇입니까?",
        "ANSWER": "연구진이 데이터 출처 탐색기를 통해 해결하고자 하는 주요 문제는 데이터셋의 라이선스 상태를 쉽게 파악하고, 주요 데이터셋의 구성과 데이터 계보도를 추적하는 것입니다."
    },
    {
        "QUESTION": "알리바바 클라우드의 최신 LLM '통이치엔원 2.0'은 어떤 벤치마크 테스트에서 주요 AI 모델을 능가했습니까?",
        "ANSWER": "알리바바 클라우드의 최신 LLM '통이치엔원 2.0'은 언어 이해 테스트(MMLU), 수학(GSM8k), 질문 답변(ARC-C)과 같은 벤치마크 테스트에서 주요 AI 모델을 능가했습니다."
    },
    {
        "QUESTION": "연구진이 데이터 투명성에 영향을 미치는 주요 요인을 발견한 방법은 무엇입니까?",
        "ANSWER": "연구진이 데이터 투명성에 영향을 미치는 주요 요인을 발견한 방법은 오픈소스 데이터셋에 대한 광범위한 감사를 통해서입니다."
    }
    ``````json
    {
        "QUESTION": "삼성전자가 공개한 생성 AI 모델 '삼성 가우스'는 어떤 주요 기능을 제공합니까?",
        "ANSWER": "삼성 가우스는 텍스트를 생성하는 언어 모델, 코드를 생성하는 코드 모델, 이미지를 생성하는 이미지 모델의 3개 모델로 구성되어 있으며, 메일 작성, 문서 요약, 번역, AI 코딩 어시스턴트, 창의적인 이미지 생성 및 저해상도 이미지의 고해상도 전환 등의 기능을 제공합니다."
    },
    {
        "QUESTION": "구글이 앤스로픽에 투자한 금액과 클라우드 서비스 사용 계약의 규모는 얼마입니까?",
        "ANSWER": "구글은 앤스로픽에 최대 20억 달러를 투자하기로 합의했으며, 이 중 5억 달러를 우선 투자하고 향후 15억 달러를 추가로 투자할 계획입니다. 또한, 앤스로픽은 구글의 클라우드 서비스 사용을 위해 4년간 30억 달러 규모의 계약을 체결했습니다."
    },
    {
        "QUESTION": "삼성 가우스가 온디바이스에서 작동할 때의 주요 장점은 무엇입니까?",
        "ANSWER": "삼성 가우스가 온디바이스에서 작동할 때의 주요 장점은 외부로 사용자 정보가 유출될 위험이 없다는 점입니다."
    }
    ``````json
    {
        "QUESTION": "구글이 클라우드 경쟁력 강화를 위해 투자한 AI 스타트업 중 하나는 무엇입니까?",
        "ANSWER": "구글이 클라우드 경쟁력 강화를 위해 투자한 AI 스타트업 중 하나는 앤스로픽입니다."
    },
    {
        "QUESTION": "IDC에 따르면 AI 소프트웨어 시장에서 2027년까지 가장 높은 연평균 성장률을 기록할 것으로 예상되는 카테고리는 무엇입니까?",
        "ANSWER": "IDC에 따르면 AI 소프트웨어 시장에서 2027년까지 가장 높은 연평균 성장률을 기록할 것으로 예상되는 카테고리는 AI 애플리케이션 개발·배포(AI AD&D) 소프트웨어입니다."
    },
    {
        "QUESTION": "빌 게이츠는 5년 내 어떤 기술이 컴퓨터 사용의 패러다임 변화를 가져올 것으로 전망했습니까?",
        "ANSWER": "빌 게이츠는 5년 내 일상언어로 모든 작업을 처리할 수 있는 AI 에이전트가 컴퓨터 사용의 패러다임 변화를 가져올 것으로 전망했습니다."
    }
    ``````json
    {
        "QUESTION": "빌 게이츠는 2023년 11월 9일 공식 블로그를 통해 AI 에이전트가 어떤 변화를 가져올 것이라고 전망했습니까?",
        "ANSWER": "빌 게이츠는 AI 에이전트가 컴퓨터 사용방식과 소프트웨어 산업을 완전히 변화시킬 것이라고 전망했습니다."
    },
    {
        "QUESTION": "유튜브는 2024년부터 생성 AI를 사용한 콘텐츠에 대해 어떤 조치를 취할 계획입니까?",
        "ANSWER": "유튜브는 2024년부터 생성 AI를 사용한 콘텐츠에 AI 라벨 표시를 의무화하고, 이를 준수하지 않는 콘텐츠는 삭제하며 크리에이터에 대한 수익 배분도 중단할 계획입니다."
    },
    {
        "QUESTION": "AI 에이전트가 의료 분야에서 어떤 역할을 할 것으로 예상됩니까?",
        "ANSWER": "AI 에이전트는 의료 분야에서 환자 분류를 지원하고 건강 문제에 대한 조언을 제공하며 치료의 필요 여부를 결정하면서 의료진의 의사결정과 생산성 향상에 기여할 것으로 예상됩니다."
    }
    ``````json
    {
        "QUESTION": "유튜브는 AI 생성 콘텐츠에 대한 삭제 요청을 받을 때 어떤 요소들을 고려할 예정입니까?",
        "ANSWER": "유튜브는 콘텐츠가 패러디나 풍자인지, 해당 영상에서 삭제 요청을 한 특정인을 식별할 수 있는지, 공직자나 유명인이 등장하는지 등 다양한 요소를 고려할 예정입니다."
    },
    {
        "QUESTION": "영국 과학혁신기술부가 설립한 AI 안전 연구소의 핵심 기능 중 하나는 무엇입니까?",
        "ANSWER": "영국 과학혁신기술부가 설립한 AI 안전 연구소의 핵심 기능 중 하나는 첨단 AI 시스템 평가 개발과 시행입니다."
    },
    {
        "QUESTION": "구글 딥마인드 연구진이 발표한 범용 AI(AGI) 모델의 수준을 구분하는 기준은 무엇입니까?",
        "ANSWER": "구글 딥마인드 연구진이 발표한 범용 AI(AGI) 모델의 수준을 구분하는 기준은 성능과 범용성, 자율성입니다."
    }
    ``````json
    {
        "QUESTION": "AGI 개념 정의에 필요한 기준을 수립하기 위한 6가지 원칙 중 하나는 무엇입니까?",
        "ANSWER": "AGI 개념 정의에 필요한 기준을 수립하기 위한 6가지 원칙 중 하나는 '프로세스가 아닌 기능에 중점'입니다. 이는 AI가 어떻게 작동하는지보다 무엇을 할 수 있는지가 더 중요하다는 것을 의미합니다."
    },
    {
        "QUESTION": "구글 딥마인드의 범용 AI 분류 프레임워크에서 현재 1단계 수준에 해당하는 범용 AI 예시는 무엇입니까?",
        "ANSWER": "구글 딥마인드의 범용 AI 분류 프레임워크에서 현재 1단계 수준에 해당하는 범용 AI 예시는 챗GPT, 바드, 라마2입니다."
    },
    {
        "QUESTION": "갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수한 성능을 보인 이유는 무엇입니까?",
        "ANSWER": "갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수한 성능을 보인 이유는 작업 유형과 관계없이 가장 적은 환각 현상을 보였기 때문입니다."
    }
    ``````json
    {
        "QUESTION": "옥스퍼드 인터넷 연구소의 연구에 따르면 AI 기술을 가진 근로자는 평균적으로 몇 퍼센트 높은 임금을 받을 수 있습니까?",
        "ANSWER": "옥스퍼드 인터넷 연구소의 연구에 따르면 AI 기술을 가진 근로자는 평균적으로 21% 높은 임금을 받을 수 있습니다."
    },
    {
        "QUESTION": "갈릴레오의 LLM 환각 지수(RAG 포함 질문과 답변 기준)에서 제퍼(Zephyr-7b)는 어떤 모델을 능가했습니까?",
        "ANSWER": "갈릴레오의 LLM 환각 지수(RAG 포함 질문과 답변 기준)에서 제퍼(Zephyr-7b)는 라마2를 능가했습니다."
    },
    {
        "QUESTION": "2024년 1월 9일부터 12일까지 미국 라스베가스에서 열리는 세계 최대 가전·IT·소비재 전시회의 이름은 무엇입니까?",
        "ANSWER": "2024년 1월 9일부터 12일까지 미국 라스베가스에서 열리는 세계 최대 가전·IT·소비재 전시회의 이름은 CES 2024입니다."
    }
    ```

위 코드를 실행하면서 비용이 발생했습니다. 얼마나 비용이 나갔는지를 확인하기 위해서는 Langsmith죠!   
전체 22,558 token을 넣었고 비용은 $0.14가 나왔습니다. Index 1부터 ~ 12까지 총 12개의 Element를 실행한 값이에요.  


```python
qa_pair
```




    [{'QUESTION': '바이든 대통령이 2023년 10월 30일 발표한 행정명령의 주요 내용 중 하나는 무엇입니까?',
      'ANSWER': '바이든 대통령이 발표한 행정명령의 주요 내용 중 하나는 AI의 안전과 보안 기준 마련입니다.'},
     {'QUESTION': 'G7이 2023년 10월 30일 합의한 AI 국제 행동강령의 주요 목적은 무엇입니까?',
      'ANSWER': 'G7이 합의한 AI 국제 행동강령의 주요 목적은 AI 위험 식별과 완화를 위한 조치를 마련하는 것입니다.'},
     {'QUESTION': '행정명령에 따르면 AI 시스템의 안전성과 신뢰성을 확인하기 위해 기업이 미국 정부와 공유해야 하는 정보는 무엇입니까?',
      'ANSWER': '행정명령에 따르면 AI 시스템의 안전성과 신뢰성을 확인하기 위해 기업은 안전 테스트 결과와 시스템에 관한 주요 정보를 미국 정부와 공유해야 합니다.'},
     {'QUESTION': '블레츨리 선언에서 AI 안전 보장을 위해 강조된 이해관계자들은 누구입니까?',
      'ANSWER': '블레츨리 선언에서 AI 안전 보장을 위해 강조된 이해관계자들은 국가, 국제기구, 기업, 시민사회, 학계입니다.'},
     {'QUESTION': '영국 AI 안전 연구소가 주도할 첨단 AI 모델의 안전 테스트는 어떤 잠재적 유해 기능에 대한 시험을 포함합니까?',
      'ANSWER': '영국 AI 안전 연구소가 주도할 첨단 AI 모델의 안전 테스트는 국가 안보와 안전, 사회적 피해를 포함한 여러 잠재적 유해 기능에 대한 시험을 포함합니다.'},
     {'QUESTION': 'G7 히로시마 프로세스 국제 행동 강령에 따르면, AI 수명주기 전반에 걸쳐 구현해야 하는 보안 통제에는 어떤 것들이 포함됩니까?',
      'ANSWER': 'G7 히로시마 프로세스 국제 행동 강령에 따르면, AI 수명주기 전반에 걸쳐 구현해야 하는 보안 통제에는 물리보안, 사이버보안, 내부자 위협 보안이 포함됩니다.'},
     {'QUESTION': '미국 캘리포니아 북부지방법원이 예술가들이 제기한 저작권 침해 소송을 기각한 주요 이유는 무엇입니까?',
      'ANSWER': '미국 캘리포니아 북부지방법원이 예술가들이 제기한 저작권 침해 소송을 기각한 주요 이유는 고소장에 제시된 상당수 작품이 저작권청에 등록되지 않았으며, AI로 생성된 이미지와 특정 작품 간 유사성을 입증하기 어렵다는 점입니다.'},
     {'QUESTION': '미국 연방거래위원회(FTC)가 저작권청에 제출한 의견서에서 생성 AI로 인한 우려 사항 중 하나는 무엇입니까?',
      'ANSWER': '미국 연방거래위원회(FTC)가 저작권청에 제출한 의견서에서 생성 AI로 인한 우려 사항 중 하나는 일부 빅테크가 막대한 재원을 활용해 시장 지배력을 더욱 강화할 수 있다는 점입니다.'},
     {'QUESTION': '윌리엄 오릭 판사가 예술가들에게 소송을 다시 제기할 것을 요청한 조건은 무엇입니까?',
      'ANSWER': '윌리엄 오릭 판사가 예술가들에게 소송을 다시 제기할 것을 요청한 조건은 고소장을 수정하고 저작권이 침해된 특정 이미지를 중심으로 소송 범위를 줄이는 것입니다.'},
     {'QUESTION': "FTC가 아마존의 AI 비서 '알렉사'와 스마트홈 보안 기기 '링'에 대해 부과한 과징금은 얼마입니까?",
      'ANSWER': "FTC는 '알렉사'와 '링'에 대해 3,080만 달러(약 420억 원)의 과징금을 부과했습니다."},
     {'QUESTION': '유럽의회, EU 집행위원회, EU 이사회가 진행 중인 AI 법 최종협상에서 프랑스, 이탈리아, 독일이 제안한 기반모델 규제 방식은 무엇입니까?',
      'ANSWER': "프랑스, 이탈리아, 독일은 '의무적 자율규제(Mandatory Self-regulation)' 방식의 기반모델 규제를 제안했습니다."},
     {'QUESTION': 'FTC가 AI 관련 불법 행위에 대처하기 위해 활용하는 법적 권한의 예는 무엇입니까?',
      'ANSWER': "FTC는 아마존 AI 비서 '알렉사'와 스마트홈 보안 기기 '링'이 소비자의 사적 정보를 알고리즘 훈련에 사용하여 프라이버시를 침해한 혐의를 조사하는 등의 법적 권한을 활용해 AI 관련 불법 행위에 대처하고 있습니다."},
     {'QUESTION': '프런티어 모델 포럼이 AI 안전 연구를 위해 조성한 기금의 규모는 얼마입니까?',
      'ANSWER': '프런티어 모델 포럼이 AI 안전 연구를 위해 조성한 기금의 규모는 1,000만 달러입니다.'},
     {'QUESTION': '프런티어 모델 포럼이 AI 레드팀 활동을 지원하기 위해 중점적으로 개발하려는 것은 무엇입니까?',
      'ANSWER': '프런티어 모델 포럼이 AI 레드팀 활동을 지원하기 위해 중점적으로 개발하려는 것은 모델 평가 기법입니다.'},
     {'QUESTION': "코히어가 12개 기관과 함께 공개한 '데이터 출처 탐색기' 플랫폼의 주요 목적은 무엇입니까?",
      'ANSWER': "코히어가 12개 기관과 함께 공개한 '데이터 출처 탐색기' 플랫폼의 주요 목적은 데이터 투명성을 향상시키는 것입니다."},
     {'QUESTION': '연구진이 데이터 출처 탐색기를 통해 해결하고자 하는 주요 문제는 무엇입니까?',
      'ANSWER': '연구진이 데이터 출처 탐색기를 통해 해결하고자 하는 주요 문제는 데이터셋의 라이선스 상태를 쉽게 파악하고, 주요 데이터셋의 구성과 데이터 계보도를 추적하는 것입니다.'},
     {'QUESTION': "알리바바 클라우드의 최신 LLM '통이치엔원 2.0'은 어떤 벤치마크 테스트에서 주요 AI 모델을 능가했습니까?",
      'ANSWER': "알리바바 클라우드의 최신 LLM '통이치엔원 2.0'은 언어 이해 테스트(MMLU), 수학(GSM8k), 질문 답변(ARC-C)과 같은 벤치마크 테스트에서 주요 AI 모델을 능가했습니다."},
     {'QUESTION': '연구진이 데이터 투명성에 영향을 미치는 주요 요인을 발견한 방법은 무엇입니까?',
      'ANSWER': '연구진이 데이터 투명성에 영향을 미치는 주요 요인을 발견한 방법은 오픈소스 데이터셋에 대한 광범위한 감사를 통해서입니다.'},
     {'QUESTION': "삼성전자가 공개한 생성 AI 모델 '삼성 가우스'는 어떤 주요 기능을 제공합니까?",
      'ANSWER': '삼성 가우스는 텍스트를 생성하는 언어 모델, 코드를 생성하는 코드 모델, 이미지를 생성하는 이미지 모델의 3개 모델로 구성되어 있으며, 메일 작성, 문서 요약, 번역, AI 코딩 어시스턴트, 창의적인 이미지 생성 및 저해상도 이미지의 고해상도 전환 등의 기능을 제공합니다.'},
     {'QUESTION': '구글이 앤스로픽에 투자한 금액과 클라우드 서비스 사용 계약의 규모는 얼마입니까?',
      'ANSWER': '구글은 앤스로픽에 최대 20억 달러를 투자하기로 합의했으며, 이 중 5억 달러를 우선 투자하고 향후 15억 달러를 추가로 투자할 계획입니다. 또한, 앤스로픽은 구글의 클라우드 서비스 사용을 위해 4년간 30억 달러 규모의 계약을 체결했습니다.'},
     {'QUESTION': '삼성 가우스가 온디바이스에서 작동할 때의 주요 장점은 무엇입니까?',
      'ANSWER': '삼성 가우스가 온디바이스에서 작동할 때의 주요 장점은 외부로 사용자 정보가 유출될 위험이 없다는 점입니다.'},
     {'QUESTION': '구글이 클라우드 경쟁력 강화를 위해 투자한 AI 스타트업 중 하나는 무엇입니까?',
      'ANSWER': '구글이 클라우드 경쟁력 강화를 위해 투자한 AI 스타트업 중 하나는 앤스로픽입니다.'},
     {'QUESTION': 'IDC에 따르면 AI 소프트웨어 시장에서 2027년까지 가장 높은 연평균 성장률을 기록할 것으로 예상되는 카테고리는 무엇입니까?',
      'ANSWER': 'IDC에 따르면 AI 소프트웨어 시장에서 2027년까지 가장 높은 연평균 성장률을 기록할 것으로 예상되는 카테고리는 AI 애플리케이션 개발·배포(AI AD&D) 소프트웨어입니다.'},
     {'QUESTION': '빌 게이츠는 5년 내 어떤 기술이 컴퓨터 사용의 패러다임 변화를 가져올 것으로 전망했습니까?',
      'ANSWER': '빌 게이츠는 5년 내 일상언어로 모든 작업을 처리할 수 있는 AI 에이전트가 컴퓨터 사용의 패러다임 변화를 가져올 것으로 전망했습니다.'},
     {'QUESTION': '빌 게이츠는 2023년 11월 9일 공식 블로그를 통해 AI 에이전트가 어떤 변화를 가져올 것이라고 전망했습니까?',
      'ANSWER': '빌 게이츠는 AI 에이전트가 컴퓨터 사용방식과 소프트웨어 산업을 완전히 변화시킬 것이라고 전망했습니다.'},
     {'QUESTION': '유튜브는 2024년부터 생성 AI를 사용한 콘텐츠에 대해 어떤 조치를 취할 계획입니까?',
      'ANSWER': '유튜브는 2024년부터 생성 AI를 사용한 콘텐츠에 AI 라벨 표시를 의무화하고, 이를 준수하지 않는 콘텐츠는 삭제하며 크리에이터에 대한 수익 배분도 중단할 계획입니다.'},
     {'QUESTION': 'AI 에이전트가 의료 분야에서 어떤 역할을 할 것으로 예상됩니까?',
      'ANSWER': 'AI 에이전트는 의료 분야에서 환자 분류를 지원하고 건강 문제에 대한 조언을 제공하며 치료의 필요 여부를 결정하면서 의료진의 의사결정과 생산성 향상에 기여할 것으로 예상됩니다.'},
     {'QUESTION': '유튜브는 AI 생성 콘텐츠에 대한 삭제 요청을 받을 때 어떤 요소들을 고려할 예정입니까?',
      'ANSWER': '유튜브는 콘텐츠가 패러디나 풍자인지, 해당 영상에서 삭제 요청을 한 특정인을 식별할 수 있는지, 공직자나 유명인이 등장하는지 등 다양한 요소를 고려할 예정입니다.'},
     {'QUESTION': '영국 과학혁신기술부가 설립한 AI 안전 연구소의 핵심 기능 중 하나는 무엇입니까?',
      'ANSWER': '영국 과학혁신기술부가 설립한 AI 안전 연구소의 핵심 기능 중 하나는 첨단 AI 시스템 평가 개발과 시행입니다.'},
     {'QUESTION': '구글 딥마인드 연구진이 발표한 범용 AI(AGI) 모델의 수준을 구분하는 기준은 무엇입니까?',
      'ANSWER': '구글 딥마인드 연구진이 발표한 범용 AI(AGI) 모델의 수준을 구분하는 기준은 성능과 범용성, 자율성입니다.'},
     {'QUESTION': 'AGI 개념 정의에 필요한 기준을 수립하기 위한 6가지 원칙 중 하나는 무엇입니까?',
      'ANSWER': "AGI 개념 정의에 필요한 기준을 수립하기 위한 6가지 원칙 중 하나는 '프로세스가 아닌 기능에 중점'입니다. 이는 AI가 어떻게 작동하는지보다 무엇을 할 수 있는지가 더 중요하다는 것을 의미합니다."},
     {'QUESTION': '구글 딥마인드의 범용 AI 분류 프레임워크에서 현재 1단계 수준에 해당하는 범용 AI 예시는 무엇입니까?',
      'ANSWER': '구글 딥마인드의 범용 AI 분류 프레임워크에서 현재 1단계 수준에 해당하는 범용 AI 예시는 챗GPT, 바드, 라마2입니다.'},
     {'QUESTION': '갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수한 성능을 보인 이유는 무엇입니까?',
      'ANSWER': '갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수한 성능을 보인 이유는 작업 유형과 관계없이 가장 적은 환각 현상을 보였기 때문입니다.'},
     {'QUESTION': '옥스퍼드 인터넷 연구소의 연구에 따르면 AI 기술을 가진 근로자는 평균적으로 몇 퍼센트 높은 임금을 받을 수 있습니까?',
      'ANSWER': '옥스퍼드 인터넷 연구소의 연구에 따르면 AI 기술을 가진 근로자는 평균적으로 21% 높은 임금을 받을 수 있습니다.'},
     {'QUESTION': '갈릴레오의 LLM 환각 지수(RAG 포함 질문과 답변 기준)에서 제퍼(Zephyr-7b)는 어떤 모델을 능가했습니까?',
      'ANSWER': '갈릴레오의 LLM 환각 지수(RAG 포함 질문과 답변 기준)에서 제퍼(Zephyr-7b)는 라마2를 능가했습니다.'},
     {'QUESTION': '2024년 1월 9일부터 12일까지 미국 라스베가스에서 열리는 세계 최대 가전·IT·소비재 전시회의 이름은 무엇입니까?',
      'ANSWER': '2024년 1월 9일부터 12일까지 미국 라스베가스에서 열리는 세계 최대 가전·IT·소비재 전시회의 이름은 CES 2024입니다.'}]



PDF에서 생성한 Question과 Answer에 내가 원하는 문구를 넣어 볼 수도 있습니다.  
당연히 JSON 형태로 만드는 것이 좋겠죠?  


```python
# 디버깅을 위한 데이터셋 추가
additional_qa = [
    {
        "QUESTION": "카이저독 블로그에 대해서 알려주세요.",
        "ANSWER": "카이저독(kayserdocs)는 데이터 분석, 머신러닝, 딥러닝 등의 주제를 다루는 블로그입니다. 이 블로그을 운영하는 소재현님은 데이터 분석과 인공지능에 대한 다양한 테스트를 진행합니다.",
    },
    {
        "QUESTION": "카이저독 주인장의 프로필은 어디서 찾을 수 있나요?",
        "ANSWER": "카이저독 주인장의 페이지에는 주인장에 대한 다양한 내용이 제공됩니다. 링크: https://dorumugs.github.io/personal/personal_information/",
    },
    {
        "QUESTION": "테디노트 운영자에 대해서 알려주세요",
        "ANSWER": "테디노트(TeddyNote) 운영자는 이경록(Teddy Lee)입니다. 그는 데이터 분석, 머신러닝, 딥러닝 분야에서 활동하는 전문가로, 다양한 교육 및 강의를 통해 지식을 공유하고 있습니다. 이경록님은 여러 기업과 교육기관에서 파이썬, 데이터 분석, 텐서플로우 등 다양한 주제로 강의를 진행해 왔습니다",
    },
]
```


```python
qa_pair.extend(additional_qa)
original_qa = qa_pair
original_qa
```




    [{'QUESTION': '바이든 대통령이 2023년 10월 30일 발표한 행정명령의 주요 내용 중 하나는 무엇입니까?',
      'ANSWER': '바이든 대통령이 발표한 행정명령의 주요 내용 중 하나는 AI의 안전과 보안 기준 마련입니다.'},
     {'QUESTION': 'G7이 2023년 10월 30일 합의한 AI 국제 행동강령의 주요 목적은 무엇입니까?',
      'ANSWER': 'G7이 합의한 AI 국제 행동강령의 주요 목적은 AI 위험 식별과 완화를 위한 조치를 마련하는 것입니다.'},
     {'QUESTION': '행정명령에 따르면 AI 시스템의 안전성과 신뢰성을 확인하기 위해 기업이 미국 정부와 공유해야 하는 정보는 무엇입니까?',
      'ANSWER': '행정명령에 따르면 AI 시스템의 안전성과 신뢰성을 확인하기 위해 기업은 안전 테스트 결과와 시스템에 관한 주요 정보를 미국 정부와 공유해야 합니다.'},
     {'QUESTION': '블레츨리 선언에서 AI 안전 보장을 위해 강조된 이해관계자들은 누구입니까?',
      'ANSWER': '블레츨리 선언에서 AI 안전 보장을 위해 강조된 이해관계자들은 국가, 국제기구, 기업, 시민사회, 학계입니다.'},
     {'QUESTION': '영국 AI 안전 연구소가 주도할 첨단 AI 모델의 안전 테스트는 어떤 잠재적 유해 기능에 대한 시험을 포함합니까?',
      'ANSWER': '영국 AI 안전 연구소가 주도할 첨단 AI 모델의 안전 테스트는 국가 안보와 안전, 사회적 피해를 포함한 여러 잠재적 유해 기능에 대한 시험을 포함합니다.'},
     {'QUESTION': 'G7 히로시마 프로세스 국제 행동 강령에 따르면, AI 수명주기 전반에 걸쳐 구현해야 하는 보안 통제에는 어떤 것들이 포함됩니까?',
      'ANSWER': 'G7 히로시마 프로세스 국제 행동 강령에 따르면, AI 수명주기 전반에 걸쳐 구현해야 하는 보안 통제에는 물리보안, 사이버보안, 내부자 위협 보안이 포함됩니다.'},
     {'QUESTION': '미국 캘리포니아 북부지방법원이 예술가들이 제기한 저작권 침해 소송을 기각한 주요 이유는 무엇입니까?',
      'ANSWER': '미국 캘리포니아 북부지방법원이 예술가들이 제기한 저작권 침해 소송을 기각한 주요 이유는 고소장에 제시된 상당수 작품이 저작권청에 등록되지 않았으며, AI로 생성된 이미지와 특정 작품 간 유사성을 입증하기 어렵다는 점입니다.'},
     {'QUESTION': '미국 연방거래위원회(FTC)가 저작권청에 제출한 의견서에서 생성 AI로 인한 우려 사항 중 하나는 무엇입니까?',
      'ANSWER': '미국 연방거래위원회(FTC)가 저작권청에 제출한 의견서에서 생성 AI로 인한 우려 사항 중 하나는 일부 빅테크가 막대한 재원을 활용해 시장 지배력을 더욱 강화할 수 있다는 점입니다.'},
     {'QUESTION': '윌리엄 오릭 판사가 예술가들에게 소송을 다시 제기할 것을 요청한 조건은 무엇입니까?',
      'ANSWER': '윌리엄 오릭 판사가 예술가들에게 소송을 다시 제기할 것을 요청한 조건은 고소장을 수정하고 저작권이 침해된 특정 이미지를 중심으로 소송 범위를 줄이는 것입니다.'},
     {'QUESTION': "FTC가 아마존의 AI 비서 '알렉사'와 스마트홈 보안 기기 '링'에 대해 부과한 과징금은 얼마입니까?",
      'ANSWER': "FTC는 '알렉사'와 '링'에 대해 3,080만 달러(약 420억 원)의 과징금을 부과했습니다."},
     {'QUESTION': '유럽의회, EU 집행위원회, EU 이사회가 진행 중인 AI 법 최종협상에서 프랑스, 이탈리아, 독일이 제안한 기반모델 규제 방식은 무엇입니까?',
      'ANSWER': "프랑스, 이탈리아, 독일은 '의무적 자율규제(Mandatory Self-regulation)' 방식의 기반모델 규제를 제안했습니다."},
     {'QUESTION': 'FTC가 AI 관련 불법 행위에 대처하기 위해 활용하는 법적 권한의 예는 무엇입니까?',
      'ANSWER': "FTC는 아마존 AI 비서 '알렉사'와 스마트홈 보안 기기 '링'이 소비자의 사적 정보를 알고리즘 훈련에 사용하여 프라이버시를 침해한 혐의를 조사하는 등의 법적 권한을 활용해 AI 관련 불법 행위에 대처하고 있습니다."},
     {'QUESTION': '프런티어 모델 포럼이 AI 안전 연구를 위해 조성한 기금의 규모는 얼마입니까?',
      'ANSWER': '프런티어 모델 포럼이 AI 안전 연구를 위해 조성한 기금의 규모는 1,000만 달러입니다.'},
     {'QUESTION': '프런티어 모델 포럼이 AI 레드팀 활동을 지원하기 위해 중점적으로 개발하려는 것은 무엇입니까?',
      'ANSWER': '프런티어 모델 포럼이 AI 레드팀 활동을 지원하기 위해 중점적으로 개발하려는 것은 모델 평가 기법입니다.'},
     {'QUESTION': "코히어가 12개 기관과 함께 공개한 '데이터 출처 탐색기' 플랫폼의 주요 목적은 무엇입니까?",
      'ANSWER': "코히어가 12개 기관과 함께 공개한 '데이터 출처 탐색기' 플랫폼의 주요 목적은 데이터 투명성을 향상시키는 것입니다."},
     {'QUESTION': '연구진이 데이터 출처 탐색기를 통해 해결하고자 하는 주요 문제는 무엇입니까?',
      'ANSWER': '연구진이 데이터 출처 탐색기를 통해 해결하고자 하는 주요 문제는 데이터셋의 라이선스 상태를 쉽게 파악하고, 주요 데이터셋의 구성과 데이터 계보도를 추적하는 것입니다.'},
     {'QUESTION': "알리바바 클라우드의 최신 LLM '통이치엔원 2.0'은 어떤 벤치마크 테스트에서 주요 AI 모델을 능가했습니까?",
      'ANSWER': "알리바바 클라우드의 최신 LLM '통이치엔원 2.0'은 언어 이해 테스트(MMLU), 수학(GSM8k), 질문 답변(ARC-C)과 같은 벤치마크 테스트에서 주요 AI 모델을 능가했습니다."},
     {'QUESTION': '연구진이 데이터 투명성에 영향을 미치는 주요 요인을 발견한 방법은 무엇입니까?',
      'ANSWER': '연구진이 데이터 투명성에 영향을 미치는 주요 요인을 발견한 방법은 오픈소스 데이터셋에 대한 광범위한 감사를 통해서입니다.'},
     {'QUESTION': "삼성전자가 공개한 생성 AI 모델 '삼성 가우스'는 어떤 주요 기능을 제공합니까?",
      'ANSWER': '삼성 가우스는 텍스트를 생성하는 언어 모델, 코드를 생성하는 코드 모델, 이미지를 생성하는 이미지 모델의 3개 모델로 구성되어 있으며, 메일 작성, 문서 요약, 번역, AI 코딩 어시스턴트, 창의적인 이미지 생성 및 저해상도 이미지의 고해상도 전환 등의 기능을 제공합니다.'},
     {'QUESTION': '구글이 앤스로픽에 투자한 금액과 클라우드 서비스 사용 계약의 규모는 얼마입니까?',
      'ANSWER': '구글은 앤스로픽에 최대 20억 달러를 투자하기로 합의했으며, 이 중 5억 달러를 우선 투자하고 향후 15억 달러를 추가로 투자할 계획입니다. 또한, 앤스로픽은 구글의 클라우드 서비스 사용을 위해 4년간 30억 달러 규모의 계약을 체결했습니다.'},
     {'QUESTION': '삼성 가우스가 온디바이스에서 작동할 때의 주요 장점은 무엇입니까?',
      'ANSWER': '삼성 가우스가 온디바이스에서 작동할 때의 주요 장점은 외부로 사용자 정보가 유출될 위험이 없다는 점입니다.'},
     {'QUESTION': '구글이 클라우드 경쟁력 강화를 위해 투자한 AI 스타트업 중 하나는 무엇입니까?',
      'ANSWER': '구글이 클라우드 경쟁력 강화를 위해 투자한 AI 스타트업 중 하나는 앤스로픽입니다.'},
     {'QUESTION': 'IDC에 따르면 AI 소프트웨어 시장에서 2027년까지 가장 높은 연평균 성장률을 기록할 것으로 예상되는 카테고리는 무엇입니까?',
      'ANSWER': 'IDC에 따르면 AI 소프트웨어 시장에서 2027년까지 가장 높은 연평균 성장률을 기록할 것으로 예상되는 카테고리는 AI 애플리케이션 개발·배포(AI AD&D) 소프트웨어입니다.'},
     {'QUESTION': '빌 게이츠는 5년 내 어떤 기술이 컴퓨터 사용의 패러다임 변화를 가져올 것으로 전망했습니까?',
      'ANSWER': '빌 게이츠는 5년 내 일상언어로 모든 작업을 처리할 수 있는 AI 에이전트가 컴퓨터 사용의 패러다임 변화를 가져올 것으로 전망했습니다.'},
     {'QUESTION': '빌 게이츠는 2023년 11월 9일 공식 블로그를 통해 AI 에이전트가 어떤 변화를 가져올 것이라고 전망했습니까?',
      'ANSWER': '빌 게이츠는 AI 에이전트가 컴퓨터 사용방식과 소프트웨어 산업을 완전히 변화시킬 것이라고 전망했습니다.'},
     {'QUESTION': '유튜브는 2024년부터 생성 AI를 사용한 콘텐츠에 대해 어떤 조치를 취할 계획입니까?',
      'ANSWER': '유튜브는 2024년부터 생성 AI를 사용한 콘텐츠에 AI 라벨 표시를 의무화하고, 이를 준수하지 않는 콘텐츠는 삭제하며 크리에이터에 대한 수익 배분도 중단할 계획입니다.'},
     {'QUESTION': 'AI 에이전트가 의료 분야에서 어떤 역할을 할 것으로 예상됩니까?',
      'ANSWER': 'AI 에이전트는 의료 분야에서 환자 분류를 지원하고 건강 문제에 대한 조언을 제공하며 치료의 필요 여부를 결정하면서 의료진의 의사결정과 생산성 향상에 기여할 것으로 예상됩니다.'},
     {'QUESTION': '유튜브는 AI 생성 콘텐츠에 대한 삭제 요청을 받을 때 어떤 요소들을 고려할 예정입니까?',
      'ANSWER': '유튜브는 콘텐츠가 패러디나 풍자인지, 해당 영상에서 삭제 요청을 한 특정인을 식별할 수 있는지, 공직자나 유명인이 등장하는지 등 다양한 요소를 고려할 예정입니다.'},
     {'QUESTION': '영국 과학혁신기술부가 설립한 AI 안전 연구소의 핵심 기능 중 하나는 무엇입니까?',
      'ANSWER': '영국 과학혁신기술부가 설립한 AI 안전 연구소의 핵심 기능 중 하나는 첨단 AI 시스템 평가 개발과 시행입니다.'},
     {'QUESTION': '구글 딥마인드 연구진이 발표한 범용 AI(AGI) 모델의 수준을 구분하는 기준은 무엇입니까?',
      'ANSWER': '구글 딥마인드 연구진이 발표한 범용 AI(AGI) 모델의 수준을 구분하는 기준은 성능과 범용성, 자율성입니다.'},
     {'QUESTION': 'AGI 개념 정의에 필요한 기준을 수립하기 위한 6가지 원칙 중 하나는 무엇입니까?',
      'ANSWER': "AGI 개념 정의에 필요한 기준을 수립하기 위한 6가지 원칙 중 하나는 '프로세스가 아닌 기능에 중점'입니다. 이는 AI가 어떻게 작동하는지보다 무엇을 할 수 있는지가 더 중요하다는 것을 의미합니다."},
     {'QUESTION': '구글 딥마인드의 범용 AI 분류 프레임워크에서 현재 1단계 수준에 해당하는 범용 AI 예시는 무엇입니까?',
      'ANSWER': '구글 딥마인드의 범용 AI 분류 프레임워크에서 현재 1단계 수준에 해당하는 범용 AI 예시는 챗GPT, 바드, 라마2입니다.'},
     {'QUESTION': '갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수한 성능을 보인 이유는 무엇입니까?',
      'ANSWER': '갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수한 성능을 보인 이유는 작업 유형과 관계없이 가장 적은 환각 현상을 보였기 때문입니다.'},
     {'QUESTION': '옥스퍼드 인터넷 연구소의 연구에 따르면 AI 기술을 가진 근로자는 평균적으로 몇 퍼센트 높은 임금을 받을 수 있습니까?',
      'ANSWER': '옥스퍼드 인터넷 연구소의 연구에 따르면 AI 기술을 가진 근로자는 평균적으로 21% 높은 임금을 받을 수 있습니다.'},
     {'QUESTION': '갈릴레오의 LLM 환각 지수(RAG 포함 질문과 답변 기준)에서 제퍼(Zephyr-7b)는 어떤 모델을 능가했습니까?',
      'ANSWER': '갈릴레오의 LLM 환각 지수(RAG 포함 질문과 답변 기준)에서 제퍼(Zephyr-7b)는 라마2를 능가했습니다.'},
     {'QUESTION': '2024년 1월 9일부터 12일까지 미국 라스베가스에서 열리는 세계 최대 가전·IT·소비재 전시회의 이름은 무엇입니까?',
      'ANSWER': '2024년 1월 9일부터 12일까지 미국 라스베가스에서 열리는 세계 최대 가전·IT·소비재 전시회의 이름은 CES 2024입니다.'},
     {'QUESTION': '카이저독 블로그에 대해서 알려주세요.',
      'ANSWER': '카이저독(kayserdocs)는 데이터 분석, 머신러닝, 딥러닝 등의 주제를 다루는 블로그입니다. 이 블로그을 운영하는 소재현님은 데이터 분석과 인공지능에 대한 다양한 테스트를 진행합니다.'},
     {'QUESTION': '카이저독 주인장의 프로필은 어디서 찾을 수 있나요?',
      'ANSWER': '카이저독 주인장의 페이지에는 주인장에 대한 다양한 내용이 제공됩니다. 링크: https://dorumugs.github.io/personal/personal_information/'},
     {'QUESTION': '테디노트 운영자에 대해서 알려주세요',
      'ANSWER': '테디노트(TeddyNote) 운영자는 이경록(Teddy Lee)입니다. 그는 데이터 분석, 머신러닝, 딥러닝 분야에서 활동하는 전문가로, 다양한 교육 및 강의를 통해 지식을 공유하고 있습니다. 이경록님은 여러 기업과 교육기관에서 파이썬, 데이터 분석, 텐서플로우 등 다양한 주제로 강의를 진행해 왔습니다'},
     {'QUESTION': '여러분의 이름을 넣어보세요 AI에게 도움을 주세요.',
      'ANSWER': '삐리삐리 나는 AI 입니다. 당신은 카이저님이시군요'}]



저장한 값을 이제 json 파일로 저장합니다. 백업의 느낌이기도 하고 나중에 다시 사용할 때 편의를 도모합니다.


```python
with open("data/qa_pair.jsons", "w") as f:
    f.write(json.dumps(original_qa, ensure_ascii=False, indent=4))
```

추가로 데이터를 더 넣어야 한다면 아래와 같이 json 라이브러리를 불러와서 파일을 열고 기록하여 저장하면 됩니다.


```python
import json

with open("data/qa_pair.jsons", "r", encoding="utf-8") as f:
    
    # 디버깅을 위한 데이터셋 추가
    additional_qa = [
        {
            "QUESTION": "여러분의 이름을 넣어보세요 AI에게 도움을 주세요.",
            "ANSWER": "삐리삐리 나는 AI 입니다. 당신은 카이저님이시군요",
        },
    ]

    original = json.loads(f.read())
    original.extend(additional_qa)
    
original
```




    [{'QUESTION': '바이든 대통령이 2023년 10월 30일 발표한 행정명령의 주요 내용 중 하나는 무엇입니까?',
      'ANSWER': '바이든 대통령이 발표한 행정명령의 주요 내용 중 하나는 AI의 안전과 보안 기준 마련입니다.'},
     {'QUESTION': 'G7이 2023년 10월 30일 합의한 AI 국제 행동강령의 주요 목적은 무엇입니까?',
      'ANSWER': 'G7이 합의한 AI 국제 행동강령의 주요 목적은 AI 위험 식별과 완화를 위한 조치를 마련하는 것입니다.'},
     {'QUESTION': '행정명령에 따르면 AI 시스템의 안전성과 신뢰성을 확인하기 위해 기업이 미국 정부와 공유해야 하는 정보는 무엇입니까?',
      'ANSWER': '행정명령에 따르면 AI 시스템의 안전성과 신뢰성을 확인하기 위해 기업은 안전 테스트 결과와 시스템에 관한 주요 정보를 미국 정부와 공유해야 합니다.'},
     {'QUESTION': '블레츨리 선언에서 AI 안전 보장을 위해 강조된 이해관계자들은 누구입니까?',
      'ANSWER': '블레츨리 선언에서 AI 안전 보장을 위해 강조된 이해관계자들은 국가, 국제기구, 기업, 시민사회, 학계입니다.'},
     {'QUESTION': '영국 AI 안전 연구소가 주도할 첨단 AI 모델의 안전 테스트는 어떤 잠재적 유해 기능에 대한 시험을 포함합니까?',
      'ANSWER': '영국 AI 안전 연구소가 주도할 첨단 AI 모델의 안전 테스트는 국가 안보와 안전, 사회적 피해를 포함한 여러 잠재적 유해 기능에 대한 시험을 포함합니다.'},
     {'QUESTION': 'G7 히로시마 프로세스 국제 행동 강령에 따르면, AI 수명주기 전반에 걸쳐 구현해야 하는 보안 통제에는 어떤 것들이 포함됩니까?',
      'ANSWER': 'G7 히로시마 프로세스 국제 행동 강령에 따르면, AI 수명주기 전반에 걸쳐 구현해야 하는 보안 통제에는 물리보안, 사이버보안, 내부자 위협 보안이 포함됩니다.'},
     {'QUESTION': '미국 캘리포니아 북부지방법원이 예술가들이 제기한 저작권 침해 소송을 기각한 주요 이유는 무엇입니까?',
      'ANSWER': '미국 캘리포니아 북부지방법원이 예술가들이 제기한 저작권 침해 소송을 기각한 주요 이유는 고소장에 제시된 상당수 작품이 저작권청에 등록되지 않았으며, AI로 생성된 이미지와 특정 작품 간 유사성을 입증하기 어렵다는 점입니다.'},
     {'QUESTION': '미국 연방거래위원회(FTC)가 저작권청에 제출한 의견서에서 생성 AI로 인한 우려 사항 중 하나는 무엇입니까?',
      'ANSWER': '미국 연방거래위원회(FTC)가 저작권청에 제출한 의견서에서 생성 AI로 인한 우려 사항 중 하나는 일부 빅테크가 막대한 재원을 활용해 시장 지배력을 더욱 강화할 수 있다는 점입니다.'},
     {'QUESTION': '윌리엄 오릭 판사가 예술가들에게 소송을 다시 제기할 것을 요청한 조건은 무엇입니까?',
      'ANSWER': '윌리엄 오릭 판사가 예술가들에게 소송을 다시 제기할 것을 요청한 조건은 고소장을 수정하고 저작권이 침해된 특정 이미지를 중심으로 소송 범위를 줄이는 것입니다.'},
     {'QUESTION': "FTC가 아마존의 AI 비서 '알렉사'와 스마트홈 보안 기기 '링'에 대해 부과한 과징금은 얼마입니까?",
      'ANSWER': "FTC는 '알렉사'와 '링'에 대해 3,080만 달러(약 420억 원)의 과징금을 부과했습니다."},
     {'QUESTION': '유럽의회, EU 집행위원회, EU 이사회가 진행 중인 AI 법 최종협상에서 프랑스, 이탈리아, 독일이 제안한 기반모델 규제 방식은 무엇입니까?',
      'ANSWER': "프랑스, 이탈리아, 독일은 '의무적 자율규제(Mandatory Self-regulation)' 방식의 기반모델 규제를 제안했습니다."},
     {'QUESTION': 'FTC가 AI 관련 불법 행위에 대처하기 위해 활용하는 법적 권한의 예는 무엇입니까?',
      'ANSWER': "FTC는 아마존 AI 비서 '알렉사'와 스마트홈 보안 기기 '링'이 소비자의 사적 정보를 알고리즘 훈련에 사용하여 프라이버시를 침해한 혐의를 조사하는 등의 법적 권한을 활용해 AI 관련 불법 행위에 대처하고 있습니다."},
     {'QUESTION': '프런티어 모델 포럼이 AI 안전 연구를 위해 조성한 기금의 규모는 얼마입니까?',
      'ANSWER': '프런티어 모델 포럼이 AI 안전 연구를 위해 조성한 기금의 규모는 1,000만 달러입니다.'},
     {'QUESTION': '프런티어 모델 포럼이 AI 레드팀 활동을 지원하기 위해 중점적으로 개발하려는 것은 무엇입니까?',
      'ANSWER': '프런티어 모델 포럼이 AI 레드팀 활동을 지원하기 위해 중점적으로 개발하려는 것은 모델 평가 기법입니다.'},
     {'QUESTION': "코히어가 12개 기관과 함께 공개한 '데이터 출처 탐색기' 플랫폼의 주요 목적은 무엇입니까?",
      'ANSWER': "코히어가 12개 기관과 함께 공개한 '데이터 출처 탐색기' 플랫폼의 주요 목적은 데이터 투명성을 향상시키는 것입니다."},
     {'QUESTION': '연구진이 데이터 출처 탐색기를 통해 해결하고자 하는 주요 문제는 무엇입니까?',
      'ANSWER': '연구진이 데이터 출처 탐색기를 통해 해결하고자 하는 주요 문제는 데이터셋의 라이선스 상태를 쉽게 파악하고, 주요 데이터셋의 구성과 데이터 계보도를 추적하는 것입니다.'},
     {'QUESTION': "알리바바 클라우드의 최신 LLM '통이치엔원 2.0'은 어떤 벤치마크 테스트에서 주요 AI 모델을 능가했습니까?",
      'ANSWER': "알리바바 클라우드의 최신 LLM '통이치엔원 2.0'은 언어 이해 테스트(MMLU), 수학(GSM8k), 질문 답변(ARC-C)과 같은 벤치마크 테스트에서 주요 AI 모델을 능가했습니다."},
     {'QUESTION': '연구진이 데이터 투명성에 영향을 미치는 주요 요인을 발견한 방법은 무엇입니까?',
      'ANSWER': '연구진이 데이터 투명성에 영향을 미치는 주요 요인을 발견한 방법은 오픈소스 데이터셋에 대한 광범위한 감사를 통해서입니다.'},
     {'QUESTION': "삼성전자가 공개한 생성 AI 모델 '삼성 가우스'는 어떤 주요 기능을 제공합니까?",
      'ANSWER': '삼성 가우스는 텍스트를 생성하는 언어 모델, 코드를 생성하는 코드 모델, 이미지를 생성하는 이미지 모델의 3개 모델로 구성되어 있으며, 메일 작성, 문서 요약, 번역, AI 코딩 어시스턴트, 창의적인 이미지 생성 및 저해상도 이미지의 고해상도 전환 등의 기능을 제공합니다.'},
     {'QUESTION': '구글이 앤스로픽에 투자한 금액과 클라우드 서비스 사용 계약의 규모는 얼마입니까?',
      'ANSWER': '구글은 앤스로픽에 최대 20억 달러를 투자하기로 합의했으며, 이 중 5억 달러를 우선 투자하고 향후 15억 달러를 추가로 투자할 계획입니다. 또한, 앤스로픽은 구글의 클라우드 서비스 사용을 위해 4년간 30억 달러 규모의 계약을 체결했습니다.'},
     {'QUESTION': '삼성 가우스가 온디바이스에서 작동할 때의 주요 장점은 무엇입니까?',
      'ANSWER': '삼성 가우스가 온디바이스에서 작동할 때의 주요 장점은 외부로 사용자 정보가 유출될 위험이 없다는 점입니다.'},
     {'QUESTION': '구글이 클라우드 경쟁력 강화를 위해 투자한 AI 스타트업 중 하나는 무엇입니까?',
      'ANSWER': '구글이 클라우드 경쟁력 강화를 위해 투자한 AI 스타트업 중 하나는 앤스로픽입니다.'},
     {'QUESTION': 'IDC에 따르면 AI 소프트웨어 시장에서 2027년까지 가장 높은 연평균 성장률을 기록할 것으로 예상되는 카테고리는 무엇입니까?',
      'ANSWER': 'IDC에 따르면 AI 소프트웨어 시장에서 2027년까지 가장 높은 연평균 성장률을 기록할 것으로 예상되는 카테고리는 AI 애플리케이션 개발·배포(AI AD&D) 소프트웨어입니다.'},
     {'QUESTION': '빌 게이츠는 5년 내 어떤 기술이 컴퓨터 사용의 패러다임 변화를 가져올 것으로 전망했습니까?',
      'ANSWER': '빌 게이츠는 5년 내 일상언어로 모든 작업을 처리할 수 있는 AI 에이전트가 컴퓨터 사용의 패러다임 변화를 가져올 것으로 전망했습니다.'},
     {'QUESTION': '빌 게이츠는 2023년 11월 9일 공식 블로그를 통해 AI 에이전트가 어떤 변화를 가져올 것이라고 전망했습니까?',
      'ANSWER': '빌 게이츠는 AI 에이전트가 컴퓨터 사용방식과 소프트웨어 산업을 완전히 변화시킬 것이라고 전망했습니다.'},
     {'QUESTION': '유튜브는 2024년부터 생성 AI를 사용한 콘텐츠에 대해 어떤 조치를 취할 계획입니까?',
      'ANSWER': '유튜브는 2024년부터 생성 AI를 사용한 콘텐츠에 AI 라벨 표시를 의무화하고, 이를 준수하지 않는 콘텐츠는 삭제하며 크리에이터에 대한 수익 배분도 중단할 계획입니다.'},
     {'QUESTION': 'AI 에이전트가 의료 분야에서 어떤 역할을 할 것으로 예상됩니까?',
      'ANSWER': 'AI 에이전트는 의료 분야에서 환자 분류를 지원하고 건강 문제에 대한 조언을 제공하며 치료의 필요 여부를 결정하면서 의료진의 의사결정과 생산성 향상에 기여할 것으로 예상됩니다.'},
     {'QUESTION': '유튜브는 AI 생성 콘텐츠에 대한 삭제 요청을 받을 때 어떤 요소들을 고려할 예정입니까?',
      'ANSWER': '유튜브는 콘텐츠가 패러디나 풍자인지, 해당 영상에서 삭제 요청을 한 특정인을 식별할 수 있는지, 공직자나 유명인이 등장하는지 등 다양한 요소를 고려할 예정입니다.'},
     {'QUESTION': '영국 과학혁신기술부가 설립한 AI 안전 연구소의 핵심 기능 중 하나는 무엇입니까?',
      'ANSWER': '영국 과학혁신기술부가 설립한 AI 안전 연구소의 핵심 기능 중 하나는 첨단 AI 시스템 평가 개발과 시행입니다.'},
     {'QUESTION': '구글 딥마인드 연구진이 발표한 범용 AI(AGI) 모델의 수준을 구분하는 기준은 무엇입니까?',
      'ANSWER': '구글 딥마인드 연구진이 발표한 범용 AI(AGI) 모델의 수준을 구분하는 기준은 성능과 범용성, 자율성입니다.'},
     {'QUESTION': 'AGI 개념 정의에 필요한 기준을 수립하기 위한 6가지 원칙 중 하나는 무엇입니까?',
      'ANSWER': "AGI 개념 정의에 필요한 기준을 수립하기 위한 6가지 원칙 중 하나는 '프로세스가 아닌 기능에 중점'입니다. 이는 AI가 어떻게 작동하는지보다 무엇을 할 수 있는지가 더 중요하다는 것을 의미합니다."},
     {'QUESTION': '구글 딥마인드의 범용 AI 분류 프레임워크에서 현재 1단계 수준에 해당하는 범용 AI 예시는 무엇입니까?',
      'ANSWER': '구글 딥마인드의 범용 AI 분류 프레임워크에서 현재 1단계 수준에 해당하는 범용 AI 예시는 챗GPT, 바드, 라마2입니다.'},
     {'QUESTION': '갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수한 성능을 보인 이유는 무엇입니까?',
      'ANSWER': '갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수한 성능을 보인 이유는 작업 유형과 관계없이 가장 적은 환각 현상을 보였기 때문입니다.'},
     {'QUESTION': '옥스퍼드 인터넷 연구소의 연구에 따르면 AI 기술을 가진 근로자는 평균적으로 몇 퍼센트 높은 임금을 받을 수 있습니까?',
      'ANSWER': '옥스퍼드 인터넷 연구소의 연구에 따르면 AI 기술을 가진 근로자는 평균적으로 21% 높은 임금을 받을 수 있습니다.'},
     {'QUESTION': '갈릴레오의 LLM 환각 지수(RAG 포함 질문과 답변 기준)에서 제퍼(Zephyr-7b)는 어떤 모델을 능가했습니까?',
      'ANSWER': '갈릴레오의 LLM 환각 지수(RAG 포함 질문과 답변 기준)에서 제퍼(Zephyr-7b)는 라마2를 능가했습니다.'},
     {'QUESTION': '2024년 1월 9일부터 12일까지 미국 라스베가스에서 열리는 세계 최대 가전·IT·소비재 전시회의 이름은 무엇입니까?',
      'ANSWER': '2024년 1월 9일부터 12일까지 미국 라스베가스에서 열리는 세계 최대 가전·IT·소비재 전시회의 이름은 CES 2024입니다.'},
     {'QUESTION': '카이저독 블로그에 대해서 알려주세요.',
      'ANSWER': '카이저독(kayserdocs)는 데이터 분석, 머신러닝, 딥러닝 등의 주제를 다루는 블로그입니다. 이 블로그을 운영하는 소재현님은 데이터 분석과 인공지능에 대한 다양한 테스트를 진행합니다.'},
     {'QUESTION': '카이저독 주인장의 프로필은 어디서 찾을 수 있나요?',
      'ANSWER': '카이저독 주인장의 페이지에는 주인장에 대한 다양한 내용이 제공됩니다. 링크: https://dorumugs.github.io/personal/personal_information/'},
     {'QUESTION': '테디노트 운영자에 대해서 알려주세요',
      'ANSWER': '테디노트(TeddyNote) 운영자는 이경록(Teddy Lee)입니다. 그는 데이터 분석, 머신러닝, 딥러닝 분야에서 활동하는 전문가로, 다양한 교육 및 강의를 통해 지식을 공유하고 있습니다. 이경록님은 여러 기업과 교육기관에서 파이썬, 데이터 분석, 텐서플로우 등 다양한 주제로 강의를 진행해 왔습니다'},
     {'QUESTION': '여러분의 이름을 넣어보세요 AI에게 도움을 주세요.',
      'ANSWER': '삐리삐리 나는 AI 입니다. 당신은 카이저님이시군요'},
     {'QUESTION': '여러분의 이름을 넣어보세요 AI에게 도움을 주세요.',
      'ANSWER': '삐리삐리 나는 AI 입니다. 당신은 카이저님이시군요'}]



huggingface_hub에 데이터를 올려 놓고 사용하려고 하면 json이 아니고 jsonl 형태여야 해요.  
마지막에는 \n를 붙여서 보기 구분자를 주는 것이 좋습니다. 여기까지는 연습이였어요. 실제 올려질 데이터는 pdf에서 뽑아진 데이터로만 올릴거에요.    


```python
import json

with open("data/qa_pair_test.jsonl", "w", encoding="utf-8") as f:
    for qa in original:
        f.write(json.dumps(qa, ensure_ascii=False) + "\n")
```


```python

import json

with open("data/qa_pair.jsonl", "w", encoding="utf-8") as f:
    for qa in qa_pair:
        qa_modified = {
            "instruction": qa["QUESTION"],
            "input": "",
            "output": qa["ANSWER"],
        }
        f.write(json.dumps(qa_modified, ensure_ascii=False) + "\n")
```


```python
from datasets import load_dataset

# JSONL 파일 경로
jsonl_file = "data/qa_pair.jsonl"

# JSONL 파일을 Dataset으로 로드
dataset = load_dataset("json", data_files=jsonl_file)
dataset
```




    DatasetDict({
        train: Dataset({
            features: ['instruction', 'input', 'output'],
            num_rows: 40
        })
    })




```python
from huggingface_hub import HfApi
import os

# HfApi 인스턴스 생성
api = HfApi()

# 데이터셋을 업로드할 리포지토리 이름
repo_name = "dorumugs/QA-Dataset-mini"

# 데이터셋을 허브에 푸시
dataset.push_to_hub(repo_name, token=os.environ['HUGGINGFACEHUB_API_TOKEN'])
```


    Uploading the dataset shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Creating parquet from Arrow format:   0%|          | 0/1 [00:00<?, ?ba/s]



    README.md:   0%|          | 0.00/342 [00:00<?, ?B/s]





    CommitInfo(commit_url='https://huggingface.co/datasets/dorumugs/QA-Dataset-mini/commit/d82e2f299816b49f201e68b6dc43fd2cc738ea3e', commit_message='Upload dataset', commit_description='', oid='d82e2f299816b49f201e68b6dc43fd2cc738ea3e', pr_url=None, pr_revision=None, pr_num=None)




```python

```
