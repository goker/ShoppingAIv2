import streamlit as st
import json
from apify_client import ApifyClient
import openai
import urllib
import os
import re
import requests
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI,OpenAIChat
from langchain.chains import LLMChain, SimpleSequentialChain

Open_AI_Key = os.getenv('OPENAI_API_KEY')
APIFY_Key = os.getenv('APIFY_API_KEY')
ASIN_Key= os.getenv('ASIN_API_KEY')

def expand_url(url):
    session = requests.Session()  # so connections are recycled
    resp = session.head(url, allow_redirects=True)
    return(resp.url)

def get_product_details(asin):
    # set up the request parameters
    params = {
    'api_key': {st.session_state['ASIN_API_KEY']},
    'amazon_domain': 'amazon.com',
    'asin': asin,
    'type': 'product'
    }

    # make the http GET request to ASIN Data API
    api_result = requests.get('https://api.asindataapi.com/request', params)
    return api_result.json()

def strip_amazon_link(link):
    parsed_url = urllib.parse.urlparse(link)
    scheme = parsed_url.scheme
    netloc = parsed_url.netloc
    path = parsed_url.path
    query_params = urllib.parse.parse_qs(parsed_url.query)
    
    # Retrieve the ASIN from the query parameters
    asin = query_params.get('ASIN', [''])[0]

    # If ASIN is not found in the query parameters, try to extract it from the path
    if not asin:
        path = parsed_url.path
        match = re.search(r'/dp/(\w+)/?', path)
        if match:
            asin = match.group(1)

    return {'stripped_url':f"{scheme}://{netloc}{path}",'asin':asin}

def generate_reviews2(text):
    openai.api_key  = st.session_state['OPENAI_API_KEY']
    prompt = f"""
       You will be given a text where each of the item seperated by | and are reviews of a product, write top 3 important bullet items for pros and cons about it and give overall score using sentimental analysis the score should be out of 100 where 0 is negative and 100 is positive. Explain the details about sentimental analysis and why you gave that score
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
        "proxyConfiguration": { "useApifyProxy": True },
        "extendedOutputFunction": "($) => { return {} }",
    }
    # Run the actor and wait for it to finish
    run = client.actor("junglee/amazon-reviews-scraper").call(run_input=run_input,timeout_secs=60)
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
    
    if(Open_AI_Key is None or APIFY_Key is None):
        with st.sidebar:
            openapikey=st.text_input('Open AI API Key')
            scapperapikey=st.text_input('Scrapper API Key')
            asinapikey=st.text_input('ASIN API Key')
            if st.button("Set") :
                st.session_state['OPENAI_API_KEY']=openapikey
                st.session_state['APIFY_API_KEY']=scapperapikey
                st.session_state['ASIN_API_KEY']=asinapikey
    else:
        st.session_state['OPENAI_API_KEY']=Open_AI_Key
        st.session_state['APIFY_API_KEY']=APIFY_Key
        st.session_state['ASIN_API_KEY']=ASIN_Key
    product_url=st.text_input('Paste Amazon Product URL')
    if ( st.button('Get Reviews')):
        product_url=expand_url(product_url)
        [col1,col2]=st.columns([1,1])
        with col1:
            with st.spinner(text="Getting Product ..."):
                stripped_link=strip_amazon_link(product_url)
                product_json = get_product_details(stripped_link['asin'])
                st.markdown(product_json['product']['title'])
                st.markdown(product_json['product']['feature_bullets_flat'])
                st.image(product_json['product']['images'][0]['link'],use_column_width='auto')
                st.markdown(product_json['product']['link']+"?tag=pixelitem0a-20")
        with col2:
            # card("Product",'',product_json['product']['images'][0],product_json['product']['link'])
            with st.spinner(text="Getting Reviews ..."):
                items_array=scraproduct(stripped_link['stripped_url'])
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
            # with st.spinner(text="AI in Process ..."):
            #     review_descriptions = "|".join([obj["reviewDescription"] for obj in items_array])
            #     response=""
            #     if len(review_descriptions) <= 10000:
            #         response=generate_why_buy(review_descriptions)
            #     else:
            #         response=generate_why_buy(review_descriptions[:10000])
            #     st.markdown(response)
        

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