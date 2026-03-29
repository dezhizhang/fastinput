import torch
import config
from model import FastInputModel
from src.dataset import get_dataloader
from predict import predict_batch


def evaluate(model, test_dataloader, device):
    """
    模型评估
    :param model:
    :param test_dataloader:
    :param device:
    :return:
    """

    top1_acc_count = 0
    top5_acc_count = 0
    total_count = 0



    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)

        targets = targets.to(device)


        top5_indexes_list = predict_batch(model, inputs)

        for target,top5_indexes in zip(targets, top5_indexes_list):
            top1_acc_count += 1
            if target == top5_indexes[0]:
                top1_acc_count += 1

            if target in top5_indexes:
                top5_acc_count += 1


    return top1_acc_count / total_count, top5_acc_count / total_count





def run_evaluate():
    """模型评估"""
    # 1. 确定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. 加载词表
    with open(config.MODELS_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f.readlines()]

    # 3. 加载模型
    model = FastInputModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))

    # 4. 获取数据集
    test_dataloader = get_dataloader()

    # 5. 评估模型
    top1_acc, top5_acc = evaluate(model, test_dataloader, device)


    print("评估结果")
    print(f"top1_acc:{top1_acc}")
    print(f"top5_acc:{top5_acc}")


if __name__ == "__main__":
    run_evaluate()



