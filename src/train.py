import torch
from torch._C.cpp import nn

from dataset import get_dataloader
from model import FastInputModel
import config

def train():
    """训练模型"""

    # 1. 确定训练设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 获取数据
    dataloader = get_dataloader()

    # 3. 加载词表
    with open(config.MODELS_DIR / 'vocab.txt', "r", encoding="utf-8") as f:
       vocab_list = [line.strip() for line in f.readlines()]

    print(vocab_list[0:10])


    # 4. 准备模型
    model = FastInputModel(vocab_size=len(vocab_list))

    # 5. 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

    # 开始训练




if __name__ == '__main__':
    train()

