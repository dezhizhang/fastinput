import torch
from dataset import get_dataloader
from model import FastInputModel
import config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个轮次
    :param model: 模型
    :param dataloader:数据集
    :param loss_fn: 损失函数
    :param optimizer:优化器
    :param device:设备
    :return:平均loss
    """
    model.train()
    total_loss = 0

    for inputs,targets in tqdm(dataloader,desc="训练"):
        inputs = inputs.to(device)
        target = targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = loss_fn(outputs, target)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    """训练模型"""

    # 1. 确定训练设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 获取数据
    dataloader = get_dataloader()

    # 3. 加载词表
    with open(config.MODELS_DIR / 'vocab.txt', "r", encoding="utf-8") as f:
        vocab_list = [line.strip() for line in f.readlines()]

    # 4. 准备模型
    model = FastInputModel(vocab_size=len(vocab_list))

    # 5. 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


    writer = SummaryWriter(log_dir=config.LOGS_DIR)

    # 开始训练
    best_loss = float("inf")

    for epoch in range(config.EPOCHS):
        print("=" * 10, f"EPOCH {epoch}", "=" * 10)

        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        # print(f"loss:{loss}")

        # 记录训练结果
        writer.add_scalar("loss", loss, epoch)

        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict())


    writer.close()

    # 保存模型







if __name__ == '__main__':
    train()
