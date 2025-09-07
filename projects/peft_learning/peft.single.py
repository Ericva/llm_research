import torch
from peft_util import get_loader, get_model
from peft import PeftConfig, PeftModel
from modelscope import BertConfig, BertForSequenceClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_, _, loader = get_loader()
model, _, _ = get_model()

model.save_pretrained('model/save_pretrained')

print(model.classifier)

# for name, module in model.named_modules():
#    if isinstance(module, torch.nn.Linear):
#        print(name, type(module))
print("---------------------------------------------------------------")

from peft import LoraConfig, TaskType, get_peft_model, LoftQConfig

config = LoraConfig(
    # 任务类型, SEQ_CLS,SEQ_2_SEQ_LM,CAUSAL_LM,TOKEN_CLS,QUESTION_ANS,FEATURE_EXTRACTION
    task_type=TaskType.SEQ_CLS,
    # 是否是推理模式.
    inference_mode=False,
    # 降秩矩阵的尺寸,这个参数会影响训练的参数量
    r=8,
    # lora的缩放系数,不影响参数量
    lora_alpha=32,
    # 降秩矩阵的dropout
    lora_dropout=0.1,
    # 指定要对原模型中的那一部分添加lora层,默认是qk线性层
    target_modules=['key'],
    # modules_to_save=['classifier']
)

model = get_peft_model(model, config)

model.print_trainable_parameters()

print(model.classifier)
print("---------------------------------------------------------------")

import datetime

# 正常训练
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.to(device)

now = datetime.datetime.now()
for i, data in enumerate(loader):
    for k, v in data.items():
        data[k] = v.to(device)
    out = model(**data)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    if i % 1 == 0:
        labels = data['labels']
        logits = out['logits'].argmax(1)
        acc = (labels == logits).sum().item() / len(labels)

        print(i, len(loader), out.loss.item(), acc)

    datetime.datetime.now()

    print("---------------------------------------------------------------")
    # peft保存,保存的文件会很小,因为只保存了lora层
    model.save_pretrained('model/peft.save_pretrained')
    print(model.base_model.classifier.modules_to_save.default.weight.size())

    print("---------------------------------------------------------------")
    # 重启初始化原模型
    model = BertForSequenceClassification.from_pretrained('model/save_pretrained')

    # 加载保存的config
    PeftConfig.from_pretrained('model/peft.save_pretrained')

    # 插入保存的lora层
    model = PeftModel.from_pretrained(model,
                                      './model/peft.save_pretrained',
                                      is_trainable=True)

    print(model.base_model.classifier.modules_to_save.default.weight.size())

    print("---------------------------------------------------------------")


    # 测试模型性能
    def test(model):
        model.to(device)
        data = next(iter(loader))
        for k, v in data.items():
            data[k] = v.to(device)
            print(len(data[k]))
        with torch.no_grad():
            outs = model(**data)
        acc = (outs.logits.argmax(1) == data.labels).sum().item() / len(
            data.labels)
        return acc


    print(test(model))

    print("---------------------------------------------------------------")
    # 合并lora层到原始模型中,效果不会改变
    model_merge = model.merge_and_unload()
    print(type(model_merge), test(model_merge))