from sklearn.svm import SVC, LinearSVC
import numpy as np
import pickle

class TemporalFilteringSVM:
    '''
    时域滤波支持向量机

    由时域滤波器和支持向量机两部分组成。支持向量机对脉冲数据的时域滤波特征进行识别

    '''
    def __init__(self, filter, **svm_kwargs):
        '''

        :param filter: 时域滤波器
        :param penalty: SVM参数。具体参见sklearn.svm.LinearSVC
        :param loss: SVM参数。具体参见sklearn.svm.LinearSVC
        :param dual: SVM参数。具体参见sklearn.svm.LinearSVC
        :param tol: SVM参数。具体参见sklearn.svm.LinearSVC
        :param C: SVM参数。具体参见sklearn.svm.LinearSVC
        :param multi_class: SVM参数。具体参见sklearn.svm.LinearSVC
        :param fit_intercept: SVM参数。具体参见sklearn.svm.LinearSVC
        :param intercept_scaling: SVM参数。具体参见sklearn.svm.LinearSVC
        :param class_weight: SVM参数。具体参见sklearn.svm.LinearSVC
        :param verbose: SVM参数。具体参见sklearn.svm.LinearSVC
        :param random_state: SVM参数。具体参见sklearn.svm.LinearSVC
        :param max_iter: SVM参数。具体参见sklearn.svm.LinearSVC
        '''
        self.filter = filter
        self.svm = LinearSVC(**svm_kwargs)

    def extract_feature(self, data):
        '''
        提取滤波特征

        :param data: 脉冲数据
        :return: 滤波后特征
        '''
        assert(isinstance(data, np.ndarray))
        n_samples = data.shape[0]
        features = self.filter(data)
        return features.reshape(n_samples, -1)

    def fit(self, train_data, train_label):
        '''
        支持向量机拟合

        :param train_data: 训练数据
        :param train_label: 训练标签
        :return: 支持向量机
        '''
        train_feature = self.extract_feature(train_data)
        self.svm.fit(train_feature, train_label)
        return self.svm

    def predict(self, test_data):
        '''
        使用支持向量机预测

        :param test_data: 测试数据
        :return: 预测结果
        '''
        test_feature = self.extract_feature(test_data)
        pred = self.svm.predict(test_feature)
        return pred
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            res = pickle.dump(self.svm, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.svm = pickle.load(f)