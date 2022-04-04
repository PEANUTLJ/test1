#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
LSTMCRF
======
A class for something.
"""
import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sequence_labeling.data_loader import Data_Loader
from sequence_labeling.data_processor import Data_Processor
from sequence_labeling.utils.evaluate import Evaluator


torch.manual_seed(1)

def argmax(vec):  # 得到每行最大值索引idx
    _, idx = torch.max(vec, 1)
    return idx.item()  # 返回每行最大值位置索引

def prepare_sequence(seq, to_ix):  # 将序列中的字转化为数字(int)表示
    idx = [to_ix[c] for c in seq]
    return torch.tensor(idx, dtype=torch.long)

# 前向算法是不断积累之前的结果，这样就会有个缺点
# 指数和积累到一定程度之后，会超过计算机浮点值的最大值
# 变成inf，这样取log后也是inf
# 为了避免这种情况，用一个合适的值clip=max去提指数和的公因子
# 这样不会使某项变得过大而无法计算
def log_sum_exp(vec):  # vec：形似[[tag个元素]]
    max_score = vec[0, argmax(vec)]  # 取vec中最大值
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # vec.size()[1]:tag数
    # 里面先做减法，减去最大值可以避免e的指数次，计算机上溢
    # 等同于torch.log(torch.sum(torch.exp(vec)))，防止e的指数导致计算机上溢
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class LSTMCRF1(nn.Module):

    # 初始化参数
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, **kwargs):
        super(LSTMCRF1, self).__init__()
        self.embedding_dim = embedding_dim  # 词嵌入维度
        self.hidden_dim = hidden_dim  # BiLSTM 隐藏层维度
        self.vocab_size = vocab_size  # 词典的大小
        self.tag_to_ix = tag_to_ix # tag到数字的映射
        self.tagset_size = len(tag_to_ix)  # tag个数
        # num_embeddings (int)：vocab_size 词典的大小
        # embedding_dim (int)：embedding_dim 嵌入向量的维度，即用多少维来表示一个符号
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # 默认使用偏置，默认不用dropout
        # 隐藏层设定为指定维度的一半，便于后期拼接
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)  # 设定为单层双向
        # 将BiLSTM提取的特征向量映射到特征空间，即经过全连接得到发射分数
        # in_features: hidden_dim 每个输入样本的大小
        # out_features:tagset_size 每个输出样本的大小
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # 转移矩阵的参数初始化，transition[i,j]代表的是从第j个tag转移到第i个tag的转移分数
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # 初始化所有其他tag转移到START_TAG的分数非常小，即不可能由其他tag转移到START_TAG
        # 初始化STOP_TAG转移到所有其他的分数非常小，即不可能有STOP_TAG转移到其他tag
        # CRF的转移矩阵,T[i,j]表示从j标签转移到i标签，
        self.transitions.data[tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, tag_to_ix["<STOP>"]] = -10000
        # 初始化LSTM的参数
        self.hidden = self.init_hidden()

    # 使用随机正态分布初始化LSTM的h0和c0
    # 否则模型自动初始化为零值，维度为[num_layers*num_directions, batch_size, hidden_dim]
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    # 计算归一化因子Z(x)
    def _forward_alg(self, feats):
        '''
        输入:发射矩阵(emission score)，实际上就是LSTM的输出
        sentence的每个word经BiLSTM后对应于每个label的得分
        输出:所有可能路径得分之和/归一化因子/配分函数/Z(x)
        '''
        # 通过前向算法递推计算
        # 初始化1行 tagset_size列的嵌套列表
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # 初始化step 0 即START位置的发射分数，"<START>"取0其他位置取-10000
        init_alphas[0][self.tag_to_ix["<START>"]] = 0.
        # 包装到一个变量里面以便自动反向传播
        forward_var = init_alphas
        # 迭代整个句子
        # feats:形似[[....], 每个字映射到tag的发射概率，
        #        [....],
        #        [....]]
        for feat in feats:
            # 存储当前时间步下各tag得分
            alphas_t = []
            for next_tag in range(self.tagset_size):
                # 取出当前tag的发射分数(与之前时间步的tag无关)，扩展成tag维
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # 取出当前tag由之前tag转移过来的转移分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # 当前路径的分数：之前时间步分数+转移分数+发射分数
                next_tag_var = forward_var + trans_score + emit_score
                # 对当前分数取log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 更新forward_var 递推计算下一个时间步
            # torch.cat 默认按行添加
            forward_var = torch.cat(alphas_t).view(1, -1)
        # 考虑最终转移到"<STOP>"
        terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
        # 对当前分数取log-sum-exp
        scores = log_sum_exp(terminal_var)
        return scores

    # 通过BiLSTM提取特征
    def _get_lstm_features(self, sentence):
        # 初始化LSTM的h0和c0
        self.hidden = self.init_hidden()
        # 使用之前构造的词嵌入为语句中每个词（word_id）生成向量表示
        # 并将shape改为[seq_len, 1(batch_size), embedding_dim]
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # LSTM网络根据输入的词向量和初始状态h0和c0
        # 计算得到输出结果lstm_out和最后状态hn和cn
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # 转换为词 - 标签([seq_len, tagset_size])表
        # 可以看作为每个词被标注为对应标签的得分情况，即维特比算法中的发射矩阵
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # 计算一个tag序列路径的得分
    def _score_sentence(self, feats, tags):
        # feats发射分数矩阵
        # 计算给定tag序列的分数，即一条路径的分数
        score = torch.zeros(1)
        # tags前面补上一个句首标签便于计算转移得分
        tags = torch.cat([torch.tensor([self.tag_to_ix["<START>"]], dtype=torch.long), tags])
        # 循环用于计算给定tag序列的分数
        for i, feat in enumerate(feats):
            # 递推计算路径分数：转移分数+发射分数
            # T[i,j]表示j转移到i
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # 加上转移到句尾的得分，便得到了gold_score
        score = score + self.transitions[self.tag_to_ix["<STOP>"], tags[-1]]
        return score

    # veterbi解码，得到最优tag序列
    def _viterbi_decode(self, feats):
        '''
        :param feats: 发射分数矩阵
        :return:
        '''
        # 便于之后回溯最优路径
        backpointers = []
        # 初始化viterbi的forward_var变量
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix["<START>"]] = 0
        # forward_var表示每个标签的前向状态得分，即上一个词被打作每个标签的对应得分值
        forward_var = init_vvars
        # 遍历每个时间步时的发射分数
        for feat in feats:
            # 记录当前词对应每个标签的最优转移结点
            # 保存当前时间步的回溯指针
            bptrs_t = []
            # 与bptrs_t对应，记录对应的最优值
            # 保存当前时间步的viterbi变量
            viterbivars_t = []
            # 遍历每个标签，求得当前词被打作每个标签的得分
            # 并将其与当前词的发射矩阵feat相加，得到当前状态，即下一个词的前向状态
            for next_tag in range(self.tagset_size):
                # transitions[next_tag]表示每个标签转移到next_tag的转移得分
                # forward_var表示每个标签的前向状态得分，即上一个词被打作每个标签的对应得分值
                # 二者相加即得到当前词被打作next_tag的所有可能得分
                # 维特比算法记录最优路径时只考虑上一步的分数以及上一步的tag转移到当前tag的转移分数
                # 并不取决于当前的tag发射分数
                next_tag_var = forward_var + self.transitions[next_tag]
                # 得到上一个可能的tag到当前tag中得分最大值的tag位置索引id
                best_tag_id = argmax(next_tag_var)
                # 将最优tag的位置索引存入bptrs_t
                bptrs_t.append(best_tag_id)
                # 添加最优tag位置索引对应的值
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 更新forward_var = 当前词的发射分数feat + 前一个最优tag当前tag的状态下的得分
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            # 回溯指针记录当前时间步各个tag来源前一步的最优tag
            backpointers.append(bptrs_t)
        # forward_var表示每个标签的前向状态得分
        # 加上转移到句尾标签STOP_TAG的转移得分
        terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
        # 得到标签STOP_TAG前一个时间步的最优tag位置索引
        best_tag_id = argmax(terminal_var)
        # 得到标签STOP_TAG当前最优tag对应的分数值
        path_score = terminal_var[0][best_tag_id]
        # 根据过程中存储的转移路径结点，反推最优转移路径
        # 通过回溯指针解码出最优路径
        best_path = [best_tag_id]
        # best_tag_id作为线头，反向遍历backpointers找到最优路径
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 去除"<START>"
        start = best_path.pop()
        # 最初的转移结点一定是人为构建的"<START>"，删除，并根据这一点确认路径正确性
        assert start == self.tag_to_ix["<START>"]
        # 最后将路径倒序即得到从头开始的最优转移路径best_path
        best_path.reverse()
        return path_score, best_path

    # 损失函数loss
    def neg_log_likelihood(self, sentence, tags):
        # 得到句子对应的发射分数矩阵
        feats = self._get_lstm_features(sentence)
        # 通过前向算法得到归一化因子Z(x)
        forward_score = self._forward_alg(feats)
        # 得到tag序列的路径得分
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    # 输入语句序列得到最佳tag路径及其得分
    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # 从BiLSTM获得发射分数矩阵
        lstm_feats = self._get_lstm_features(sentence)
        # 使用维特比算法进行解码，计算最佳tag路径及其得分
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def run_Model(self, model, op_mode):
        if self.criterion_name not in self.criterion_dict:
            raise ValueError("There is no criterion_name: {}.".format(self.criterion_name))
        loss_function = self.criterion_dict[self.criterion_name]()
        if self.optimizer_name not in self.optimizer_dict:
            raise ValueError("There is no optimizer_name: {}.".format(self.optimizer_name))
        optimizer = self.optimizer_dict[self.optimizer_name](model.parameters(), lr=self.learning_rate)

        if op_mode == 'train':
            model.train()
            for epoch in range(self.num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
                for sentenceList, tagList in Data_Loader().data_generator(batch_size=self.batch_size, op_mode=op_mode):
                    for i in range(len(sentenceList)):
                        batch_x = torch.LongTensor(sentenceList[i])
                        batch_y = torch.LongTensor(tagList[i])
                        model.zero_grad()
                        tag_scores = model(batch_x)
                        loss = loss_function(tag_scores, batch_y)
                        loss.backward()
                        optimizer.step()

            model_save_path = self.data_root + self.model_save_path
            torch.save(model, '{}'.format(model_save_path))
        elif op_mode == 'eval' or 'test':
            model.eval()
        else:
            print("op_mode参数未赋值(train/eval/test)")

        with torch.no_grad():
            for sentenceList, tagList in Data_Loader().data_generator(op_mode=op_mode):
                batch_x = torch.LongTensor(sentenceList[0])
                print(batch_x)
                scores = model(batch_x)

                # print("After Train:", scores)
                # print('标注结果转换为Tag索引序列：', torch.max(scores, dim=1))
                y_predict = list(torch.max(scores, dim=1)[1].numpy())

                # 将索引序列转换为Tag序列
                y_pred = self.index_to_tag(y_predict)
                y_true = self.index_to_tag(tagList[0])

                # 输出评价结果
                print(y_pred)
                print(y_true)
                print(Evaluator().acc(y_true, y_pred))


    def index_to_tag(self, y):
        tag_index_dict = Data_Processor().load_tags()
        index_tag_dict = dict(zip(tag_index_dict.values(), tag_index_dict.keys()))
        y_tagseq = []
        for i in range(len(y)):
            y_tagseq.append(index_tag_dict[y[i]])
        return y_tagseq


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')