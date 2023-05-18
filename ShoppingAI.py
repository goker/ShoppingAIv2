import streamlit as st
import json
from apify_client import ApifyClient
import openai
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI,OpenAIChat
from langchain.chains import LLMChain, SimpleSequentialChain

# Open_AI_Key = os.getenv('OPENAI_API_KEY')
# APIFY_Key = os.getenv('APIFY_API_KEY')

def generate_reviews2(text):
    openai.api_key  = st.session_state['OPENAI_API_KEY']
    prompt = f"""
       You will be given a text where each of the item seperated by | and are reviews of a product, write top 3 important bullet items for pros and cons about it and give overall score using sentimental analysis the score should be out of 100 where 0 is negative and 100 is positive.
        Text:" {text} "
        """
    print(prompt)
    return get_completion(prompt,temp=0.2)

def get_completion(prompt, model="gpt-3.5-turbo",temp=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

@st.cache_data
def scraproduct(product_url):
    client = ApifyClient(st.session_state['APIFY_API_KEY'])
    # Prepare the actor input
    run_input = {
        "productUrls": [{ "url": product_url }],
        "maxReviews": 30,
        "timeout" :60,
        "proxyConfiguration": { "useApifyProxy": True },
        "extendedOutputFunction": "($) => { return {} }",
    }
    # Run the actor and wait for it to finish
    run = client.actor("junglee/amazon-reviews-scraper").call(run_input=run_input)
    print(run)
    print("*********")
    items_array = [item for item in client.dataset(run["defaultDatasetId"]).iterate_items()]
    return items_array

def outputjson(items_array):
        # Serializing json
    json_object = json.dumps(items_array, indent=4)
    
    # Writing to sample.json
    with open("review.json", "w") as outfile:
        outfile.write(json_object)

def generate_reviews(text):
    openai.api_key  = st.session_state['OPENAI_API_KEY']
    llm = OpenAIChat(temperature=0.2,model="gpt-3.5-turbo",max_tokens=500)
    prompt = PromptTemplate(
    input_variables=["text"],
    template="""
       You will be given a text where each of the item seperated by | and are reviews of a product, write top 3 most important bullet items for pros and cons.
        Text:" {text} "
        """,
    )
    # chain = SimpleSequentialChain(llm=llm, prompt=prompt,verbose=True)
    chain = LLMChain(llm=llm, prompt=prompt,verbose=True)
    return chain.run(text)

def generate_why_buy(text):
    openai.api_key  = st.session_state['OPENAI_API_KEY']
    llm = OpenAIChat(temperature=0.2,model="gpt-3.5-turbo",max_tokens=500)
    prompt = PromptTemplate(
    input_variables=["text"],
    template="""
       You will be given a text where each of the item seperated by | and are reviews of a product, Generate 3 items why should a customer buy and 3 items for not to buy.
        Text:" {text} "
        """,
    )
    # chain = SimpleSequentialChain(llm=llm, prompt=prompt,verbose=True)
    chain = LLMChain(llm=llm, prompt=prompt,verbose=True)
    return chain.run(text)

def main():
    with st.sidebar:
        openapikey=st.text_input('Open AI API Key')
        scapperapikey=st.text_input('Scrapper API Key')
        if st.button("Set") :
            st.session_state['OPENAI_API_KEY']=openapikey
            st.session_state['APIFY_API_KEY']=scapperapikey
    
    product_url=st.text_input('Paste Amazon Product URL')
    if ( st.button('Get Reviews')):
        with st.spinner(text="Getting Reviews ..."):
            items_array=scraproduct(product_url)
            if len(items_array):
                st.session_state['reviews']=items_array
        with st.spinner(text="AI in Process ..."):
            review_descriptions = "|".join([obj["reviewDescription"] for obj in items_array])
            response=""
            if len(review_descriptions) <= 10000:
                response=generate_reviews2(review_descriptions)
            else:
                response=generate_reviews2(review_descriptions[:10000])
            st.markdown(response)
        with st.spinner(text="AI in Process ..."):
            review_descriptions = "|".join([obj["reviewDescription"] for obj in items_array])
            response=""
            if len(review_descriptions) <= 10000:
                response=generate_why_buy(review_descriptions)
            else:
                response=generate_why_buy(review_descriptions[:10000])
            st.markdown(response)
        

if __name__ == "__main__":
    st.set_page_config(page_title="Shopping AI", layout="wide")
    st.header("Shopping Assistant AI")
    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    main()