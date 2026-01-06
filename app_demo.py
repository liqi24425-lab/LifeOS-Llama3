import gradio as gr
import torch
from unsloth import FastLanguageModel

# 加载模型 (假设你已经训练并保存了 lora_model)
# 注意：如果是首次运行，这里需要指向 base model，并加载 adapter
# 为了演示简单，这里只写推理逻辑结构

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # 这里填你保存的目录
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

# 提示词模版
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def chat(message, history):
    # 这里应该放置真正的模型推理代码
    # 为了防止你在没有GPU的本地环境报错，这里写一个模拟返回
    # 实际部署时取消下面的注释
    
    """
    inputs = tokenizer(
        [alpaca_prompt.format(message, "", "")], 
        return_tensors = "pt"
    ).to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens = 128)
    response = tokenizer.batch_decode(outputs)[0]
    return response.split("### Response:\\n")[-1].replace("<|end_of_text|>", "")
    """
    
    return f"[Mock Output] Model received: {message}. (Run on GPU to see real inference)"

# 创建界面
iface = gr.ChatInterface(
    fn=chat,
    title="🧬 LifeOS Assistant",
    description="Ask me about your schedule, health protocols, or diet.",
    examples=["What is the plan for Tuesday?", "My shoulder hurts.", "I slept poorly last night."],
    theme="soft"
)

if __name__ == "__main__":
    iface.launch()