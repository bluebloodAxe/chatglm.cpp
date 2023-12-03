from transformers import AutoModelForCausalLM, AutoTokenizer

# 下载模型和分词器
model_name = "THUDM/ChatGLM3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 使用模型生成文本
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

print(output[0])
