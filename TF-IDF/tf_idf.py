import math
import jieba.posseg as psg

#TODO这个是加载停用词表
def stopword():
    stop_word_path = r'D:\ProgrammingSoftware\pycharm\HanLp\stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,encoding='utf-8').readlines()]
    return stopword_list

#TODO先进行分词
def cut_word(sentence):
    seg_list = psg.cut(sentence)
    return seg_list

#TODO: text中的停用词过滤掉
def word_filter(seg_list):
    stopword_list = stopword()
    filter_list = []
    for seg in seg_list:
        word = seg.word
        flag = seg.flag
        if not flag.startswith('n'):
            continue
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)
    return filter_list

#TODO: 算出来filter_list中的每个词的tf值
def tf_value(filter_list):
    filter_list = filter_list
    tf_value_dict = {}
    tf_value = {}
    for word in filter_list:
        tf_value_dict[word] = tf_value_dict.get(word, 0.0) + 1.0

    for key, value in tf_value_dict.items():
        tf_value[key] = float(value / len(filter_list))
    return tf_value

#TODO 以上已经算出来了tf的值,接下来需要计算idf的值
# 1. 加载数据集
def load_data():
    corpus_path = r'D:\ProgrammingSoftware\pycharm\HanLp\corpus.txt'
    doc_list = []
    for line in open(corpus_path, 'r', encoding='utf-8'):
        content = line.strip()
        seg_list = cut_word(content)
        filter_word = word_filter(seg_list)
        doc_list.append(filter_word)
    return doc_list

# 2.进行idf的统计
def train_idf():
    doc_list = load_data()
    idf_dic = {}
    total_doc_num = len(doc_list) # 总的文档的数目
    # 每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

    # 按照idf公式进行转换
    for key, value in idf_dic.items():
        # 加1是拉普拉斯平滑,防止部分新词在语料库中没有出现导致分母为0
        idf_dic[key] = math.log(total_doc_num / (1.0 + value))
    return idf_dic

#TODO 进行tf-idf的计算

def tf_idf(tf):
    tf_value_dict = tf # tf的值,tf_value是个字典
    idf_value = train_idf() # idf的值,idf是个字典
    tf_idf_dict = {}
    for key, value in tf_value_dict.items():
        tf_idf_dict[key] = value
        for key_idf, value_idf in idf_value.items():
            if key == key_idf:
                tf_idf_dict[key] = value * value_idf
    return tf_idf_dict

# 进行排序,输出10个最高值的
def rank():
    keyword_num = 10
    tf_idf_dict = tf_idf(tf)
    final_dict = sorted(tf_idf_dict.items(), key = lambda x: x[1], reverse = True)
    for i in range(0, len(final_dict)):
        print(final_dict[i][0] + '/', end = '')
        if i > 10:
            break


if __name__ == '__main__':
    text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
    '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
    '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
    '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
    '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
    '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
    '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
    '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
    '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
    '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
    '常委会主任陈健倩介绍了大会的筹备情况。'

    seg_list = cut_word(text)
    filter_word = word_filter(seg_list)
    tf = tf_value(filter_word)
    tf_idf(tf)
    rank()
    
    
    

