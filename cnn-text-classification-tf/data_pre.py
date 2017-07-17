import jieba
import jieba.posseg as pseg
import os

flag_list = ['t','q','p','u','e','y','o','w','m']
def jiebafenci(all_the_text):
    re = ""
    relist = ""
    words = pseg.cut(all_the_text)
    count = 0
    for w in words:
        flag = w.flag  #词性
        tmp = w.word   #单词
        #print "org: "+tmp
        #\u4e00-\u9fa5：unicode编码，一种全世界语言都包括的一种编码。
        #这两个unicode值正好是Unicode表中的汉字的头和尾，用来判断是否是汉字
        if len(tmp)>1 and len(flag)>0 and flag[0] not in flag_list and  tmp[0]>=u'/u4e00' and tmp[0]<=u'\u9fa5':
            re = re + " " + w.word
    re = re.replace("\n"," ").replace("\r"," ")   
    if  len(re)>40:
        relist = re
    relist = relist + "\n"
    return relist
'''
将同一类别下的多个文本分词，合并成一个文本，方便处理
'''
def getTrainData(inpath,outfile):
    i=0
    for filename in os.listdir(inpath):
        
        fw = open(outfile+".txt","a")  
        file_object = open(inpath+"\\"+filename,'r', encoding='UTF-8')
        try:
            all_the_text = file_object.read()
            
            #all_the_text = all_the_text.decode("gb2312").encode("utf-8")
            pre_text = jiebafenci(all_the_text)
            pre_text.encode('UTF-8')
            if len(pre_text)>30:
                fw.write(pre_text)
                
        except:
            print('@'*20)
            pass
        finally:
            file_object.close()
    fw.close()

inpath = 'E:\data\SogouC.reduced\Reduced\C000013'
outfile = 'E:\data\SogouC.reduced\Reduced\C000013'
inpath1 = 'E:\data\SogouC.reduced\Reduced\C000020'
outfile1 = 'E:\data\SogouC.reduced\Reduced\C000020'
getTrainData(inpath,outfile)
getTrainData(inpath1,outfile1)