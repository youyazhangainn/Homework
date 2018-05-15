from numpy import *
from math import log



traindata = loadtxt("iris.txt",delimiter = ',',usecols = (0,1,2,3),dtype = float)
trainlabel = loadtxt("iris.txt",delimiter = ',',usecols = (range(4,5)),dtype = str)
feaname = ["#0","#1","#2","#3"] # 定义四个属性

def calentropy(label):
    n = label.size # 数据集的大小

    count = {}
    for curlabel in label:
        if curlabel not in count.keys():
            count[curlabel] = 0
        count[curlabel] += 1
    entropy = 0
    #print count
    for key in count:
        pxi = float(count[key])/n #notice transfering to float first
        entropy -= pxi*log(pxi,2)
    return entropy


# 通过标签"splitfea_idx"来遍历数据集
args = mean(traindata,axis = 0)
def splitdata(oridata, splitfea_idx):
    arg = args[splitfea_idx]  # 获取所有维度的平均值
    idx_less = []  # 创建新的列表，包括具有小于中心点的特征的数据
    idx_greater = []  # 包括具有大于中心点的特征的数据
    n = len(oridata)
    for idx in range(n):
        d = oridata[idx]
        if d[splitfea_idx] < arg:
            # add the newentry into newdata_less set
            idx_less.append(idx)
        else:
            idx_greater.append(idx)
    return idx_less, idx_greater



# 根据 index标注数据
def idx2data(oridata, label, splitidx, fea_idx):
    idxl = splitidx[0]  # split_less_indices
    idxg = splitidx[1]  # split_greater_indices
    datal = []
    datag = []
    labell = []
    labelg = []
    for i in idxl:
        datal.append(append(oridata[i][:fea_idx], oridata[i][fea_idx + 1:]))
    for i in idxg:
        datag.append(append(oridata[i][:fea_idx], oridata[i][fea_idx + 1:]))
    labell = label[idxl]
    labelg = label[idxg]
    return datal, datag, labell, labelg

def choosebest_splitnode(oridata, label):
    n_fea = len(oridata[0])
    n = len(label)
    base_entropy = calentropy(label)
    best_gain = -1
    for fea_i in range(n_fea):  # 计算每个分割特征下的熵
        cur_entropy = 0
        idxset_less, idxset_greater = splitdata(oridata, fea_i)
        if(len(idxset_less)<5):
            break
        prob_less = float(len(idxset_less)) / n
        prob_greater = float(len(idxset_greater)) / n
        cur_entropy += prob_less * calentropy(label[idxset_less])
        cur_entropy += prob_greater * calentropy(label[idxset_greater])
        info_gain = base_entropy - cur_entropy
        if (info_gain > best_gain):
            best_gain = info_gain
            best_idx = fea_i
    return best_idx


# 基于信息增益的决策树生成

def buildtree(oridata, label):
    if label.size == 0:  # 如果没有样本属于这个分支
        return "NULL"
    listlabel = label.tolist()
    #当此子集中的所有样本属于一个类时停止
    if listlabel.count(label[0]) == label.size:
        return label[0]
        # 如果没有额外的特征，则返回该子集中的大多数样本标签。
    if len(feaname) == 0:
        cnt = {}
        for cur_l in label:
            if cur_l not in cnt.keys():
                cnt[cur_l] = 0
            cnt[cur_l] += 1
        maxx = -1
        for keys in cnt:
            if maxx < cnt[keys]:
                maxx = cnt[keys]
                maxkey = keys
        return maxkey

    bestsplit_fea = choosebest_splitnode(oridata, label)  # 获得最佳分割特征
    print(bestsplit_fea, len(oridata[0]))
    cur_feaname = feaname[bestsplit_fea]  # 将特征名称添加到字典中
    print(cur_feaname)
    nodedict = {cur_feaname: {}}

    del(feaname[bestsplit_fea])  # 从 feaname中删除当前特征
    split_idx = splitdata(oridata, bestsplit_fea)
    data_less, data_greater, label_less, label_greater = idx2data(oridata, label, split_idx, bestsplit_fea)
    # 递归生成树，左右树分别为“<”和“>”分支。
    nodedict[cur_feaname]["<"] = buildtree(data_less, label_less)
    nodedict[cur_feaname][">"] = buildtree(data_greater, label_greater)
    return nodedict

mytree = buildtree(traindata,trainlabel)
print (mytree)
