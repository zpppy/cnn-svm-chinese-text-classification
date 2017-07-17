# cnn-libsvm-chinese-text-classification
本项目主要针对CNN和SVM在文本分类领域的效果做对比。采用搜狐语料库（https://pan.baidu.com/share/link?uk=3089318666&shareid=528982730&adapt=pc&fr=ftw）, 实验结果为：cnn的正确率为96.67%，libsvm的正确率为90%。


## 项目一：cnn文本分类：
- 主要参考[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)论文的模型结构。具体模型结构为卷积层-->激活层（ReLU）-->池化层-->（dropout）全连接层

- 每个文本看作一张图片，横轴为词向量的维度，纵轴为文本的长度

- 使用多个不同尺寸的卷积核提取文本的不同特征

- word vector如果是随机初始化的。如果有好的中文向量模型，可以先使用该模型预训练下，作为初始值，就像迁移学习的fine tuning。

- 实验环境：Python 3.5  Tensorflow=1.2.0


-----------------------------------------------------------------------------------------------------
## 项目二：libsvm文本分类：
- 实现文本分类的主要包括几个步骤文本分词处理，特征选择，特征权重计算，文本特征向量表示，基于训练文本的特征向量数据训练SVM模型，对于测试集进行特征向量表示代入训练得到的svm模型中进行预测分类

- 在此程序的实现中，使用结巴分词对文本进行分词

- 特征选择采用的是开方检验的算法,选择的特征在SVMFeature.txt文件中，每个类别选取1000个特征，10个类别10000特征，由于
重复，计算出来的特征为9508个。

- 特征权重计算，采用的是TF*IDF计算算法，训练文本的特征向量表示数据在train.svm文件中，测试文本的特征向量表示数据在test.svm中

- 对train.svm对于模型训练，和对于test.svm模型预测，使用的是LIBSVM库，链接为http://www.csie.ntu.edu.tw/~cjlin/libsvm/， 文本分类正确率为36/40=0.90。文本分类具体结果在LIBSVM文件夹中。

完整运行命令：
1. 数据处理：
python data_process.py

2. 特征选择：
python FeatureSelecion.py

3. 特征权重计算：
python FeatureWeight.py
python TestFeatureWeight.py

4. 对train.svm文件数据进行缩放到[0,1]区间
svm-scale -l 0 -u 1 train.svm > trainscale.svm

5. 对test.svm文件数据进行缩放到[0,1]区间
svm-scale -l 0 -u 1 test.svm > testscale.svm

6. 对trainscale.svm 文件进行模型训练
svm-train -s 1 trainscale.svm trainscale.model

7. 对testscale.svm 文件进行模型预测，得到预测结果，控制台会输出正确率
svm-predict testscale.svm trainscale.model testscale.result
---------------------------------------------------------------------------
更多文本分类的方法和技术可参考：
https://zhuanlan.zhihu.com/p/25928551
