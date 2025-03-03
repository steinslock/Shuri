import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
# 设置 TensorBoard 记录器
writer = SummaryWriter('logs')

#torch.cuda.set_device(3)

# 加载CSV文件
data = pd.read_csv('/home/qiangminc/codes/AaaTEST/dataset_small_RK.csv')

# 检查数据
print(data.head())

# 模型和分词器的名称
#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "meta-llama/Llama-2-7b-chat-hf"
#model_name = "meta-llama/Llama-2-13b-chat-hf"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型并移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    
#npcr中使用到的mlp模型
class mlp(nn.Module):
    def __init__(self, in_f, out_f):
        super(mlp, self).__init__()
        self.layer1 = nn.Linear(in_f, 4096)
        self.active1 = nn.Tanh()
        self.layer2 = nn.Linear(4096, 4096)
        self.active2 = nn.ReLU()
        self.layer3 = nn.Linear(4096, out_f)

    def forward(self, x):
        out = self.layer1(x)
        out = self.active1(out)
        out = self.layer2(out)
        out = self.active2(out)
        out = self.layer3(out)
        return out
    
#定义NPCR模型
class npcr_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(npcr_model, self).__init__()

        self.dropout = nn.Dropout(0.5)

        #self.nn1 = nn.Linear(input_dim, hidden_dim)
        self.nn1 = mlp(input_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, output_dim, bias=False)

        self.init_weights()

    # 原始NPCR代码中用的初始化方法
    # def init_weights(self):
    #     """
    #     Here we reproduce Keras default initialization weights for consistency with Keras version
    #     """
    #     ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
    #     hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
    #     b = (param.data for name, param in self.named_parameters() if 'bias_ih' in name or 'bias_hh' in name)
    #     # nn.init.uniform(self.embed.weight.data, a=-0.5, b=0.5)
    #     for t in ih:
    #         nn.init.xavier_uniform_(t)
    #     for t in hh:
    #         nn.init.orthogonal_(t)
    #     for t in b:
    #         nn.init.constant_(t, 0)

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x0, x1):

        x0_nn1 = self.nn1(x0)
        x1_nn1 = self.nn1(x1)

        # x0_nn1 = self.dropout(x0_nn1)
        # x1_nn1 = self.dropout(x1_nn1)

        diff_x = (x0_nn1 - x1_nn1)
        y = self.output(diff_x)
        return y

# 确定LLaMA模型的隐藏层维度
example_data = """Write a concise summary of the text. Return your responses with maximum 5 sentences that cover the key points of the text.
Text: This is a photo from when I went to Okinawa for the first time with my kids. I remembered going to Shuri Castle while walking around and being held in my arms.
SUMMARY: 
"""
example_score = 2.5
example_input = tokenizer(example_data, return_tensors="pt").to(device)

if 'token_type_ids' in example_input:
    del example_input['token_type_ids']

with torch.no_grad():
    example_output = model(**example_input, output_hidden_states=True)

    ref_hidden_states = example_output.hidden_states[-1]    #获取参考文本的嵌入（隐藏层表示）
    example_input = torch.mean(ref_hidden_states, dim=1)    #取平均值作为参考输入

    hidden_size = example_output.hidden_states[-1].shape[-1]  # 获取隐藏层的维度
    print("隐藏层维度：", hidden_size)

# 初始化MLP模型并移动到GPU
input_dim = hidden_size
hidden_dim = 4096
output_dim = 3    #使用CrossEntropy loss，将文本分为3类，0类对应小于等于1分，1类对应大于1分小于等于2分，2类对应大于2分小于等于3分

npcr = npcr_model(input_dim, hidden_dim, output_dim).to(device)

# 定义损失函数和优化器
weights = torch.tensor([113/(63*3), 113/(34*3), 113/(16*3)]).to(device)  #类权重，根据每个类别的样本数量进行计算
criterion = nn.CrossEntropyLoss(weight=weights)  # 使用交叉熵损失
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #梯度裁剪
optimizer = optim.AdamW(npcr.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

# 读取回忆和分数
texts = data['memory_comment_EN'].tolist()

# 添加prompt
modified_texts = [
    f"Write a concise summary of the text. Return your responses with maximum 5 sentences that cover the key points of the text. If the text is already less then 5 sentences, do not rewrite it. Text: {text} SUMMARY: "
    for text in texts
]

labels = data['SCORE'].tolist()

# 将分数分为 3 类
labels = pd.cut(labels, bins=[-float('inf'), 1, 2, float('inf')], labels=[0, 1, 2])

# print(f"labels = {labels}")

# print(f"output_dim = {output_dim}")

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(modified_texts, labels, test_size=0.2, random_state=42)


# 训练循环
num_epochs = 5000

for epoch in range(num_epochs):
    all_preds = []
    all_labels = []
    
    for text, label in zip(texts, labels):
        label = torch.tensor([label]).long().to(device)  # 将标签转换为整数，并移动到 GPU
        #label = label.squeeze()

        #print(f"labels = {label}")
        # 使用分词器进行编码
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # 删除 token_type_ids 参数
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']

        # 获取 LLaMA2 模型的输出结果
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        # 将隐藏状态的平均值作为 npcr 的 x0 输入
        mlp_input = torch.mean(hidden_states, dim=1)

        optimizer.zero_grad()

        # 前向传播，输出 logits
        y_predict = npcr(mlp_input, example_input)

        #print(f"y_predict = {y_predict}")

        # 计算损失（分类任务）
        loss = criterion(y_predict, label)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 保存预测值和真实标签
        all_preds.append(torch.argmax(y_predict, dim=-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())
    
    writer.add_scalar('Loss/train', loss.item(), epoch)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # 计算 QWK
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    qwk_score = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    writer.add_scalar('QWK/validation', qwk_score, epoch)
    print(f"Epoch {epoch+1}/{num_epochs}, QWK: {qwk_score:.4f}")

print("训练完成")

# # 保存训练好的MLP模型权重
# torch.save(mlp.state_dict(), "path_to_save_mlp_model.pth")

# # 定义推理函数
# def predict(text):
#     # 将输入文本进行分词和编码
#     inputs = tokenizer(text, return_tensors="pt").to(device)

#     # 删除 token_type_ids 参数
#     if 'token_type_ids' in inputs:
#         del inputs['token_type_ids']

#     # 获取LLaMA2模型的输出结果
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#         hidden_states = outputs.hidden_states[-1]  # 使用最后一层的隐藏状态

#     # 将隐藏状态的平均值作为MLP的输入
#     mlp_input = torch.mean(hidden_states, dim=1)

#     # 前向传播
#     with torch.no_grad():
#         y_predict = mlp(mlp_input)
    
#     # 获取预测结果
#     _, predicted_label = torch.max(y_predict, dim=1)
#     return predicted_label.item()

# # 示例输入
# input_text = "这是一个测试输入。"
# predicted_label = predict(input_text)
# print(f"Predicted label: {predicted_label}")
