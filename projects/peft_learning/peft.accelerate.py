import torch
from tools import get_loader, get_model
from peft import PeftConfig, PeftModel
from peft import LoraConfig, TaskType, get_peft_model, LoftQConfig
from modelscope import BertConfig, BertForSequenceClassification
from accelerate import Accelerator, DeepSpeedPlugin
import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_, _, loader = get_loader(data_size=60000,batch_size=64)
model, _, _ = get_model()

config = LoraConfig(
    #任务类型, SEQ_CLS,SEQ_2_SEQ_LM,CAUSAL_LM,TOKEN_CLS,QUESTION_ANS,FEATURE_EXTRACTION
    #task_type=TaskType.SEQ_CLS,
    #是否是推理模式.
    inference_mode=False,
    #降秩矩阵的尺寸,这个参数会影响训练的参数量
    r=8,
    #lora的缩放系数,不影响参数量
    lora_alpha=32,
    #降秩矩阵的dropout
    lora_dropout=0.1,
    #指定要对原模型中的那一部分添加lora层,默认是qk线性层
    #target_modules=['classifier'],
    target_modules=['key'],
    modules_to_save=['classifier'] # 和task_type一样的功效,都是指定classifier也要被学习
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
print(model.classifier)
print("---------------------------------------------------------------")

#正常训练
optimizator = torch.optim.Adam(model.parameters(), lr=1e-4)
deepspeed = DeepSpeedPlugin(zero_stage = 2, gradient_clipping = 1.0)
accelerator = Accelerator(deepspeed_plugin = deepspeed)
##model.to(device)
model, optimizator, dataloader = accelerator.prepare(model, optimizator, loader)
now = datetime.datetime.now()

for i, data in enumerate(dataloader):
    optimizator.zero_grad()
    out = model(**data)
    accelerator.backward(out.loss)
    optimizator.step()

    ## spmd每一个都会打印,不是gather的
    #if i % 10 == 0:
    #    labels = data['labels']
    #    logits = out['logits'].argmax(1)
    #    acc = (labels == logits).sum().item() / len(labels)
    #    print(i, len(loader), out.loss.item(), acc)

    ## 使用gather写法
    with torch.no_grad():
        preds = out.logits.argmax(dim=-1)
        labels = data["labels"]
        # 跨进程收集
        preds = accelerator.gather_for_metrics(preds)
        labels = accelerator.gather_for_metrics(labels)
        acc = (preds == labels).float().mean().item()

    if accelerator.is_main_process and i % 10 == 0:
        accelerator.print(f"step {i}  loss={out.loss.item():.4f}  acc={acc:.4f}")

datetime.datetime.now()
print("---------------------------------------------------------------")

# 关键：用 get_state_dict 汇总（兼容 DDP/FSDP/ZeRO），再 unwrap 后保存
state_dict = accelerator.get_state_dict(model)
if accelerator.is_main_process:
    unwrapped = accelerator.unwrap_model(model)     # 拿到真实的 PeftModel
    unwrapped.load_state_dict(state_dict)
    unwrapped.save_pretrained('model_accelerate/peft.save_pretrained')
    print(unwrapped.base_model.classifier.modules_to_save.default.weight.size())


accelerator.save(state_dict, "model_accelerate/train_model.save.pth")
accelerator.save_state("model_accelerate/train_model.save_state.pth")