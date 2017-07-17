# cnn-libsvm-chinese-text-classification
cnn项目介绍：
## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

-----------------------------------------------------------------------------------------------------
libsvm项目介绍：
1.实现文本分类的主要包括几个步骤文本分词处理，特征选择，特征权重计算，文本特征向量表示，基于训练文本的特征向量数据训练SVM模型，对于测试集进行特征向量表示代入训练得到的svm模型中进行预测分类

2.在此程序的实现中，我使用HMM模型实现了一个分词程序(使用了Python 的numpy库)，然后对搜狗语料库中的文本进行分词处理(在这里我对搜狗预料库进行了裁剪，我只选择了2200个文本，每个类别220个文本，其中前200个文本用作训练，后20个文本用作测试)
我实现的分词程序的正确率在80%左右，具体分词正确率结果在ChineseSegmentation\PKU_GB\score.txt文件中

3.特征选择我采用的是开方检验的算法,选择的特征在SVMFeature.txt文件中，每个类别选取1000个特征，10个类别10000特征，由于
重复，计算出来的特征为9508个。

4.特征权重计算，采用的是TF*IDF计算算法，训练文本的特征向量表示数据在train.svm文件中，测试文本的特征向量表示数据在test.svm中

5.对train.svm对于模型训练，和对于test.svm模型预测，使用的是LIBSVM库，链接为http://www.csie.ntu.edu.tw/~cjlin/libsvm/，文本分类正确率为36/40=0.90。文本分类具体结果在LIBSVM文件夹中,具体测试命令在这个文件夹下的测试命令.txt文件中。

测试命令：
数据处理：
python data_process.py

特征选择：
python FeatureSelecion.py

特征权重计算：
python FeatureWeight.py
python TestFeatureWeight.py

对train.svm文件数据进行缩放到[0,1]区间

svm-scale -l 0 -u 1 train.svm > trainscale.svm

对test.svm文件数据进行缩放到[0,1]区间

svm-scale -l 0 -u 1 test.svm > testscale.svm

对trainscale.svm 文件进行模型训练

svm-train -s 1 trainscale.svm trainscale.model

对testscale.svm 文件进行模型预测，得到预测结果，控制台会输出正确率

svm-predict testscale.svm trainscale.model testscale.result
