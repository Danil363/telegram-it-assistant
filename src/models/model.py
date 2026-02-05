from transformers import AutoModelForCausalLM, AutoTokenizer
import os                            
model_name = "microsoft/DialoGPT-medium"  
model_path = "src/model_configs/checkpoint-20000"                

class DialogModel():
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(self, query: str):
        pretrained_text = self.tokenizer(query, return_tensors="pt")

        rez = self.model.generate(pretrained_text["input_ids"], max_length=50)

        return self.tokenizer.decode(rez[0], skip_special_tokens=True)


model = DialogModel(model_name)

model.predict('Привет. Как дела?')



# model = AutoModelForCausalLM.from_pretrained(model_path)
