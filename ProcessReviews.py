
import json
import openai
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI,OpenAIChat
from langchain.chains import LLMChain, SimpleSequentialChain
import requests


Open_AI_Key = os.getenv('OPENAI_API_KEY')

# Specify the path to your JSON file
json_file_path = "review.json"

def generate_reviews2(text):
    openai.api_key  = Open_AI_Key
    prompt = f"""
       You will be given a text where each of the item seperated by | and are reviews of a product, write top 3 important bullet items for pros and cons about it and give overall score using sentimental analysis the score should be out of 100 where 0 is negative and 100 is positive.
        Pros: <output json>
        Cons: <output json>
        Text:" {text} "
        """
    print(prompt)
    return get_completion(prompt,temp=0)

def get_completion(prompt, model="gpt-3.5-turbo",temp=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def generate_reviews(text):
    openai.api_key  = Open_AI_Key
    llm = OpenAIChat(temperature=0.2,model="gpt-3.5-turbo",max_tokens=500)
    prompt = PromptTemplate(
    input_variables=["text"],
    template="""
       You will be given a text where each of the item seperated by | and are reviews of a product, write top 3 important bullet items for pros and cons about it. Output Json object for pros and cons
        Pros: <output json>
        Cons: <output json>
        Text:" {text} "
        
        """,
    )
    # chain = SimpleSequentialChain(llm=llm, prompt=prompt,verbose=True)
    chain = LLMChain(llm=llm, prompt=prompt,verbose=True)
    return chain.run(text)

def main():
# Read the JSON file
    with open(json_file_path, "r") as file:
        json_data = json.load(file)
        review_descriptions = "|".join([obj["reviewDescription"] for obj in json_data])
        response=""
        if len(review_descriptions) <= 12090:
            response=generate_reviews2(review_descriptions)
        else:
            response=generate_reviews2(review_descriptions[:12090])
        print(review_descriptions)
        print(response)
        
def expand_url(url):
    session = requests.Session()  # so connections are recycled
    resp = session.head(url, allow_redirects=True)
    return(resp.url)
    

    
if __name__ == "__main__":
    expand_url("https://www.amazon.com/UooEA-Conduction-Waterproof-Swimming-Headphones/dp/B0C1V7C21J?_encoding=UTF8&pd_rd_w=fI1QB&content-id=amzn1.sym.08900ed4-dd64-49b1-bce7-2e717defb1aa&pf_rd_p=08900ed4-dd64-49b1-bce7-2e717defb1aa&pf_rd_r=FDYGNEQSB5HZNAGZYJB8&pd_rd_wg=tsLEy&pd_rd_r=a072e4e6-9571-48ce-a267-47b8a4a0f827&linkCode=sl1&tag=pixelitem0a-20&linkId=901c3ecf8017f6230549043b1de792d9&language=en_US&ref_=as_li_ss_tl")
    # main()