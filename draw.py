import torch
import hiddenlayer as hl
from  model import ActionValueNet

# 实例化模型
model = ActionValueNet()

# 准备输入尺寸，HiddenLayer 需要一个例子输入来推断每层的输出 shape
# 这里需要把两个输入合并成一个 tuple/list 传给 build_graph
state = torch.zeros(1, 685)
history = torch.zeros(1, 10, 60)   # T=10

# 因为模型 forward 接受两个参数，我们需要包装一下
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, state, history):
        return self.model(state, history)

wrapper = ModelWrapper(model)

# 构建图
hl.build_graph(wrapper, (state, history))