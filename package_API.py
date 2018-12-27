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


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# from sklearn.feature_extraction.text import TfidfVectorizer
# v = TfidfVectorizer(stop_words="english")
# import string
# no_biaodian = string.punctuation(input)
def listElement2str(list1):
    list2 = []
    for i in range(len(list1)):
        if type(list1[i]) == float:
            if isnan(list1[i]):
                list2.append("")
        else:
            list2.append(list1[i])
    return list2

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

def see_result(train_X,train_y,test_X,test_y,clf,savepath):
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


class Spam_Detector():
    def nltk_fc(self):
        """用nltk进行英文分词"""
        content_S = []
        for line in self.content:
            current_segment = nltk.word_tokenize(line)
            content_S.append(current_segment)
        self.content_S = content_S
        return self.content_S

    def drop_stopwords(self):
        """去掉停用词"""
        self.stopwords = np.array(pd.read_csv("resource_data/stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'],encoding='utf-8')).tolist()
        self.clean_words = drop_stopwords(self.content_S, self.stopwords)[0]
        self.all_words = drop_stopwords(self.content_S, self.stopwords)[1]
        return self.clean_words,self.stopwords

    def tfidf_draw_feature(self):
        """用tf-idf进行提取特征"""
        logger.info("TF-IDF is drawing the feature!")
        # （1）首先需要对每句话抽出的词进行拼接成一句子，虽然是断断续续的
        corpus = []
        for line in self.clean_words:
            line_single = " ".join(line)
            corpus.append(line_single)
        self.corpus_df = pd.DataFrame({"corpus:":corpus})
        self.corpus = corpus

        # (2)然后用tfidf模型训练（必须输入是一个list形式）
        self.vectorizer = TfidfVectorizer(analyzer='word')
        self.vectorizer.fit(self.corpus)

        self.X = self.vectorizer.transform(self.corpus).toarray()  # tfidf 矩阵
        self.diction = self.vectorizer.vocabulary_ #tfidf字典（比如 array ：111）
        return self.X,self.diction

    def train_model(self):
        """6.建立二分类模型"""
        # （1）将数据集分为训练和测试集
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.label_new, test_size=0.2, random_state=42)
        # (2)尝试不同的分类器效果朴素贝叶斯和随机森林
        self.RF_model = RandomForestClassifier(n_estimators=500, max_features=10, n_jobs=-1)
        self.RF_model.fit(self.train_X, self.train_y)
        see_result(self.train_X, self.train_y, self.test_X, self.test_y, self.RF_model, "ROC/RF_roc.jpg")
        save_model(self.RF_model, "model/RF_model.model")

    def input_one_record(self,input):
        """输入单个语句，得到输出结果"""
        self.current_singe = nltk.word_tokenize(input)
        self.end_words = []
        for i in self.current_singe:
            if i not in self.stopwords:
                self.end_words.append(i)
        self.words_join = " ".join(self.end_words)

        """3.tfidf+model"""
        tfidf_matrix = self.vectorizer.transform([self.words_join]).toarray()
        self.result = self.RF_model.predict(tfidf_matrix)
        return self.result


    def __init__(self):
        logger.info("Starting up the Spam Detector: ")
        self.path = 'resource_data/spam.csv'
        """1.加载数据"""
        logger.info("Loading the data!")
        spam_data = pd.read_csv(self.path, encoding='utf-8')
        spam_data.columns = ['label', 'content1', 'content2', 'content3', 'content4']
        col_names = spam_data.columns.values
        """2.数据整理"""
        content1 = listElement2str(np.array(spam_data["content1"]).tolist())
        content2 = listElement2str(np.array(spam_data["content2"]).tolist())
        content3 = listElement2str(np.array(spam_data["content3"]).tolist())
        content4 = listElement2str(np.array(spam_data["content4"]).tolist())
        # 整理标签
        label_orig = np.array(spam_data["label"]).tolist()
        self.label_new = []
        for i in range(len(label_orig)):
            if label_orig[i] == 'ham':
                self.label_new.append(0)
            else:
                self.label_new.append(1)
        self.content = []
        for i in range(len(content1)):
            content_single = "".join((content1[i], content2[i], content3[i], content4[i]))
            self.content.append(content_single)
        new_spam = pd.DataFrame({"label": self.label_new, "content": self.content})

        self.nltk_fc()
        self.drop_stopwords()
        self.tfidf_draw_feature()
        self.train_model()

class One_record_test():
    def __init__(self):
        # 关于单一一条记录进行测试
        """1.读取model，读取语料库"""
        self.model = load_model("model/RF_model.model")
        self.vectorizer = load_model("model/vectorizer.model")
        self.corpus = pd.read_csv("corpus.csv")
        self.spam_data = pd.read_csv('spam_new.csv', encoding='utf-8')

    def confirm_result(self,content_1_str):
        """2.分词+去停用词+合并"""
        self.current_S = nltk.word_tokenize(content_1_str)
        self.stopwords = np.array(pd.read_csv("source_data/stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'],
                                encoding='utf-8')).tolist()
        self.end_words = []
        for i in self.current_S:
            if i not in self.stopwords:
                self.end_words.append(i)

        self.words_join = " ".join(self.end_words)

        """3.tfidf+model"""
        self.tfidf_matrix = self.vectorizer.transform([self.words_join]).toarray()
        self.result = self.model.predict(self.tfidf_matrix)
        return self.result


if __name__ == '__main__':
    # obj = Spam_Detector()
    input_str = "Hello,I am YeYan!"
    # result = obj.input_one_record(input_str)
    # print(result)

    obj2 = One_record_test()
    result2 = obj2.confirm_result(input_str)
    print(result2)

