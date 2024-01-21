import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

class OpenAIModel():
    def __init__(self, engine, system_prompt = None):
        self.engine = engine
        self.system_prompt = system_prompt
    
    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return client.embeddings.create(
                input = [text], 
                model=self.engine)['data'][0]['embedding']
    
    def get_chat_completion(self, messages, max_tokens: int = 1700):
        return client.chat.completions.create(
            model=self.engine,
            messages=messages,
            max_tokens = max_tokens,
            )['choices'][0]['message']['content']

    def classify_text(self, query):        
        return client.chat.completions.create(
        model=self.engine,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        )


if __name__ == "__main__":
    system_prompt = """You are an intelligent and helpful assistant who will classify text. Your output will be a JSON that contains the category of the text and more details. Categories should be high level, like code, documentation, literature, legal, numbers, website text, etc. If you don't know the category, write N/A. Feel free to come up with your own categories. Details should give more specific information. For example, if the category is code, say what language; if it's legal, is it a warranty, contract, license, etc. Here are some examples for you.\nText:  '<link rel="stylesheet" type="text/css" href="../../../../stylesheet.css" title="Style">\n<link rel="stylesheet" type="text/css" href="../../../../jquery/jquery-ui.css" title="Style">\n<script type="text/javascript" src="../../../../script.js"></'\nYou: {'category': 'code', 'details': 'css'}. Text:  ' against all the gods of Egypt I will execute judgment: I am the LORD. And the blood shall be to you for a token upon the houses where ye are: and when I see the blood, I will pass over you, and the plague shall not be upon you to destroy you, when I smite the land'\nYou: {'category': 'literature', 'details': 'bible'}.\n"""
    
    gpt4 = OpenAIModel("gpt-4-1106-preview", system_prompt = system_prompt)
    file_path = 'data/12b'

    all_mem_12b_data = pd.read_csv(f'{file_path}/mem_evals_gen_data.csv')
    mem_12b_data = all_mem_12b_data[all_mem_12b_data['char_by_char_similarity'] == 1]

    start_time = time.time()
    classifications = []
    
    print(len(mem_12b_data['gen'].to_list()))
    print()
    
    for t in mem_12b_data['gen'].to_list():
        classification = gpt4.classify_text(f"Text: {t}\nYou:").choices[0].message.content
        print(t)
        print(classification)
        print()
        
        classifications.append(classification)
    
    print("saving")
    pickle.dump(classifications, open(f'{file_path}/labeled_mem_clusters.pkl', 'wb'))
    
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds")

#nohup python3 label_mem_clusters.py &> data/12b/label_mem_clusters.out &