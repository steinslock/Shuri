import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置 TensorBoard 记录器
writer = SummaryWriter('logs')

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        essay = self.data.iloc[idx]['memory_comment_EN']
        score = self.data.iloc[idx]['SCORE']
        return essay, score

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

        #y = torch.sigmoid((2/3)*y)

        return y

# 训练函数
def train(mlp_model, dataloader, criterion, optimizer, lr_scheduler, num_epochs, llm_model, llm_tokenizer):


    #获取参考文章分数以及嵌入

    """
    ref_essay: 作为参考的文章
    ref_score: 参考文章的分数
    ref_input: 参考文章的嵌入向量，用于后续MLP网络的输入
    hidden_size: 嵌入向量的维度，隐藏层维度

    """
    
    ref_essay = dataloader.dataset.data['memory_comment_EN'].iloc[0]
    modified_texts = [f"Write a concise summary of the text. Return your responses with maximum 5 sentences that cover the key points of the text. Text: {ref_essay} SUMMARY: "]
    ref_input = llm_tokenizer(modified_texts, return_tensors="pt").to(device)
    if 'token_type_ids' in ref_input:
        del ref_input['token_type_ids']
    
    with torch.no_grad():
        ref_output = llm_model(**ref_input, output_hidden_states=True)

        ref_hidden_states = ref_output.hidden_states[-1]    #获取参考文本的嵌入（隐藏层表示）
        ref_input = torch.mean(ref_hidden_states, dim=1)    #取平均值作为参考输入

        hidden_size = ref_output.hidden_states[-1].shape[-1]  # 获取隐藏层的维度
        print("隐藏层维度：", hidden_size)
    
    ref_score = dataloader.dataset.data['SCORE'].iloc[0]

    
    
    print(ref_score)


    for epoch in range(num_epochs):
        for essays, scores in dataloader:
            # 使用 LLM 生成特征向量
            essay_input = llm_tokenizer(essays, return_tensors="pt").to(device)
            if 'token_type_ids' in essay_input:
                del essay_input['token_type_ids']
            with torch.no_grad():
                essay_output = llm_model(**essay_input, output_hidden_states=True)

                essay_hidden_states = essay_output.hidden_states[-1]    #获取参考文本的嵌入（隐藏层表示）
                essay_input = torch.mean(essay_hidden_states, dim=1)    #取平均值作为参考输入
        
            optimizer.zero_grad()
        
            y_predict = mlp_model(essay_input, ref_input)
            y_predict = y_predict.squeeze()

            diff_score = scores - ref_score
            diff_score = diff_score.to(device)

            loss = criterion(y_predict, diff_score)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        writer.add_scalar('Loss/train', loss.item(), epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")






if __name__ == "__main__":
    # 参数设置
    input_size = 4096  # 输入特征的大小，需要根据文本向量化的结果设置
    output_size = 1
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 16

    #llm相关
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    llm_model = AutoModelForCausalLM.from_pretrained(model_name)
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)


    #mlp相关
    mlp_model = mlp(input_size, output_size)
    # 创建数据集和数据加载器
    dataset = CustomDataset('/home/qiangminc/codes/AaaTEST/dataset_small_RK.csv')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建模型、损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(mlp_model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

    llm_model = llm_model.to(device)
    mlp_model = mlp_model.to(device)
    # 训练模型
    train(mlp_model, dataloader, criterion, optimizer, lr_scheduler, num_epochs, llm_model, llm_tokenizer)