import numpy as np
import pandas as pd
from math import isnan
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
# import pandas_profiling

def listElement2str(list1):
    list2 = []
    for i in range(len(list1)):
        if type(list1[i]) == float:
            if isnan(list1[i]):
                list2.append("")
        else:
            list2.append(list1[i])
    return list2

def drop_stopwords(contents,stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        #遍历contents里面的每个元素，每一个元素都是一个list，也就是一条信息
        line_clean = []
        for word in line:
            #遍历每个list里的单词，因为之前已经经过切分了
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean,all_words

def draw_ROC_curve(y_test,y_predict,savepath):
    false_positive_rate,true_positive_rate,threshold = roc_curve(y_test,y_predict)
    roc_auc = auc(false_positive_rate,true_positive_rate)
    plt.title("ROC")
    plt.plot(false_positive_rate,true_positive_rate,'b',label = 'AUC = %0.2f'%roc_auc)
    plt.legend(loc= 'low right')
    plt.plot([0,1],[1,0],'r--')
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.savefig(savepath)
    # plt.show()
    # plt.close(0)

def see_result(train_X,trian_y,test_X,test_y,clf,savepath):
    # (2)使用交叉验证调参
    scores = cross_val_score(clf, train_X, train_y, cv=10)
    print(scores.mean())
    precision_report = classification_report(test_y, clf.predict(test_X))
    print(precision_report)
    Confusion_matrix = confusion_matrix(test_y, clf.predict(test_X))
    print(Confusion_matrix)
    draw_ROC_curve(test_y,clf.predict(test_X),savepath)

def load_model(model_path):
    # 载入模型
    model = pickle.load(open(model_path,'rb'))
    return model

def save_model(clf,model_path):
    # 保存模型
    model = pickle.dump(clf,open(model_path,'wb'))


if __name__ == '__main__':
    """1.看数据"""
    spam_data = pd.read_csv('resource_data/spam.csv',encoding='utf-8')
    print(spam_data.head())
    print(spam_data.shape)
    spam_data.columns = ['label','content1','content2','content3','content4']
    col_names = spam_data.columns.values
    print(col_names)
    # spam_profile_report = pandas_profiling.ProfileReport(spam_data)
    # spam_profile_report.to_file("spam.html")

    """2.数据整理"""
    content1 = listElement2str(np.array(spam_data["content1"]).tolist())
    content2 = listElement2str(np.array(spam_data["content2"]).tolist())
    content3 = listElement2str(np.array(spam_data["content3"]).tolist())
    content4 = listElement2str(np.array(spam_data["content4"]).tolist())
    #整理标签
    label_orig = np.array(spam_data["label"]).tolist()
    label_new = []
    for i in range(len(label_orig)):
        if label_orig[i] == 'ham':
            label_new.append(0)
        else:
            label_new.append(1)
    content = []
    for i in range(len(content1)):
        content_single = "".join((content1[i],content2[i],content3[i],content4[i]))
        content.append(content_single)
    print(content)
    new_spam = pd.DataFrame({"label":label_new,"content":content})
    new_spam.to_csv("spam_new.csv",index=False,encoding='utf-8')
    # """3.用jieba进行中文分词"""
    # content_S = []
    # for line in content:
    #     current_segment = jieba.lcut(line)
    #     if len(current_segment) > 1 and current_segment != '\r\n':  # 换行符
    #         content_S.append(current_segment)
    # print(content_S)
    """3.用nltk进行英文分词"""
    content_S = []
    for line in content:
        current_segment = nltk.word_tokenize(line)
        content_S.append(current_segment)
    print(content_S)

    """4.去掉停用词"""
    stopwords = pd.read_csv("resource_data/stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')
    stopwords.head(20)
    stopwords = np.array(stopwords["stopword"]).tolist()
    clean_words = drop_stopwords(content_S,stopwords)[0]
    all_words = drop_stopwords(content_S,stopwords)[1]
    print(clean_words)

    """5.用tf-idf进行提取特征"""
    # （1）首先需要对每句话抽出的词进行拼接成一句子，虽然是断断续续的
    corpus = []
    for line in clean_words:
        line_single = " ".join(line)
        corpus.append(line_single)
    print(corpus[0])
    corpus_df = pd.DataFrame({"corpus:":corpus})
    # corpus_df.to_csv("corpus.csv",index=False,encoding='utf-8')
    # (2)然后用tfidf（必须输入是一个list形式）
    vectorizer = TfidfVectorizer(analyzer='word')
    vectorizer.fit(corpus)
    # 保存vectorizer这个tfidf模型
    # save_model(vectorizer,"model/vectorizer.model")
    X = vectorizer.transform(corpus).toarray()  # tfidf 矩阵
    diction = vectorizer.vocabulary_ #tfidf字典（比如 array ：111）
    print(X.shape)

    """6.建立二分类模型"""
    # （1）将数据集分为训练和测试集
    train_X, test_X, train_y, test_y = train_test_split(X,label_new,test_size=0.2,random_state=42)
    # (2)尝试不同的分类器效果朴素贝叶斯和随机森林
    start_time = datetime.now()
    NB_model = GaussianNB()
    NB_model.fit(train_X,train_y)
    see_result(train_X,train_y,test_X,test_y,NB_model,"ROC/NB_roc.jpg")
    # save_model(NB_model,"model/NB_model.model")
    end_time = datetime.now()
    print("Using time is {}".format(end_time - start_time))

    start_time = datetime.now()
    RF_model = RandomForestClassifier(n_estimators=500, max_features=10, n_jobs=-1,class_weight={0:10,1:1})
    RF_model.fit(train_X, train_y)
    see_result(train_X, train_y, test_X, test_y, RF_model, "ROC/RF_roc.jpg")
    # save_model(RF_model, "model/RF_model.model")
    end_time = datetime.now()
    print("Using time is {}".format(end_time - start_time))

