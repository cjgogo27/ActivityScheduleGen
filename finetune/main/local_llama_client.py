"""
本地Llama模型推理封装
模拟OpenAI API接口,方便替换现有代码
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Message:
    """模拟OpenAI的Message对象"""
    def __init__(self, content: str):
        self.content = content


class Choice:
    """模拟OpenAI的Choice对象"""
    def __init__(self, message: Message):
        self.message = message


class ChatCompletion:
    """模拟OpenAI的ChatCompletion对象"""
    def __init__(self, choices: List[Choice]):
        self.choices = choices


class ChatCompletions:
    """模拟OpenAI的chat.completions对象"""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> ChatCompletion:
        """
        模拟OpenAI的chat.completions.create方法
        
        Args:
            model: 模型名称(本地实现中忽略此参数)
            messages: 消息列表,格式: [{"role": "system/user/assistant", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大生成token数
        
        Returns:
            ChatCompletion对象
        """
        
        # 构建Llama3.1格式的prompt
        prompt = self._build_llama_prompt(messages)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        # 只取生成的部分,去除输入的prompt
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 构建返回对象
        message = Message(content=response_text)
        choice = Choice(message=message)
        return ChatCompletion(choices=[choice])
    
    def _build_llama_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        将OpenAI格式的messages转换为Llama3.1的prompt格式
        
        Llama3.1格式:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        
        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
        
        {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        prompt = "<|begin_of_text|>"
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        # 添加assistant开始标记,准备生成
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt


class Chat:
    """模拟OpenAI的chat对象"""
    def __init__(self, model, tokenizer, device):
        self.completions = ChatCompletions(model, tokenizer, device)


class LocalLlamaClient:
    """
    本地Llama模型客户端,模拟OpenAI API
    可以直接替换OpenAI()使用
    """
    
    def __init__(
        self,
        base_model_path: str,
        lora_path: Optional[str] = None,
        device: str = "auto",
        torch_dtype = torch.bfloat16,
        **kwargs  # 兼容OpenAI的api_key, base_url, timeout等参数
    ):
        """
        初始化本地Llama客户端
        
        Args:
            base_model_path: 基础Llama模型路径
            lora_path: LoRA权重路径(如果有)
            device: 设备选择 ("auto", "cuda", "cpu")
            torch_dtype: 模型精度
            **kwargs: 兼容OpenAI客户端的其他参数(会被忽略)
        """
        logger.info("=" * 60)
        logger.info("初始化本地Llama模型客户端")
        logger.info("=" * 60)
        
        # 加载tokenizer
        logger.info(f"加载Tokenizer: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=False,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载基础模型
        logger.info(f"加载基础模型: {base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        # 如果提供了LoRA权重,则加载
        if lora_path:
            logger.info(f"加载LoRA权重: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            logger.info("  LoRA权重加载完成")
        
        self.model.eval()  # 设置为评估模式
        
        # 确定设备
        if device == "auto":
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device(device)
        
        logger.info(f"模型加载完成")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  精度: {torch_dtype}")
        logger.info("=" * 60)
        
        # 创建chat对象,模拟OpenAI接口
        self.chat = Chat(self.model, self.tokenizer, self.device)
    
    def __call__(self, *args, **kwargs):
        """兼容性方法"""
        return self


def create_local_llama_client(
    base_model_path: str = "/data/mayue/cjy/Other_method/FinalTraj/finetune/Llama/LLM-Research/Meta-Llama-3___1-8B-Instruct",
    lora_path: str = "/data/mayue/cjy/Other_method/FinalTraj/finetune/output/llama3_1_trajectory_lora/final",
    **kwargs
) -> LocalLlamaClient:
    """
    创建本地Llama客户端的便捷函数
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA权重路径
        **kwargs: 其他参数
    
    Returns:
        LocalLlamaClient实例
    """
    return LocalLlamaClient(
        base_model_path=base_model_path,
        lora_path=lora_path,
        **kwargs
    )


# ============ 测试代码 ============

def test_local_client():
    """测试本地客户端"""
    print("\n测试本地Llama客户端...\n")
    
    # 创建客户端
    client = create_local_llama_client()
    
    # 测试对话
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant specialized in generating realistic daily activity schedules."
        },
        {
            "role": "user",
            "content": """Generate a daily schedule for a full-time worker who needs to go to work and do grocery shopping.

Output in JSON format:
```json
[
  {"activity": "home", "start_time": "00:00", "end_time": "07:30"},
  {"activity": "work", "start_time": "07:30", "end_time": "17:00"}
]
```"""
        }
    ]
    
    print("发送请求...")
    response = client.chat.completions.create(
        model="llama3.1",  # 本地实现中忽略此参数
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    print("\n生成的回复:")
    print("-" * 60)
    print(response.choices[0].message.content)
    print("-" * 60)


if __name__ == "__main__":
    test_local_client()
