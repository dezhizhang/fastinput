import torch
import config
from  model import FastInputModel
from src.dataset import get_dataloader


def run_evaluate():
    """模型评估"""
    # 1. 确定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. 加载词表
    with open(config.MODELS_DIR / 'vocab.txt','r',encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f.readlines()]

    # 3. 加载模型
    model = FastInputModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))


    # 4. 获取数据集
    test_dataloader = get_dataloader()


    # 5. 评估模型
    top1_acc,top5_acc = evaluate(model, test_dataloader, device)



