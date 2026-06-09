import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from  langchain_core.prompts import load_prompt, PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st



load_dotenv()


# model="deepseek-ai/DeepSeek-V4-Pro:novita" 


hf_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
#llama_model="meta-llama/Llama-3.1-8B"
qwen_model="Qwen/Qwen2.5-7B-Instruct"

chat_model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_api_key,
    model=qwen_model 
)



st.header('LLaMA-3 Chatbot')




paper_input = st.selectbox( "Select Research Paper Name", 
                           [
                                "Attention Is All You Need", 
                                "BERT: Pre-training of Deep Bidirectional Transformers",
                                "GPT-3: Language Models are Few-Shot Learners",
                                "Diffusion Models Beat GANs on Image Synthesis"
                            ] 
                        )

style_input = st.selectbox( "Select Explanation Style",
                           [
                               "Beginner-Friendly",
                                "Technical",
                                "Code-Oriented",
                                "Mathematical"
                            ] 
                        ) 

length_input = st.selectbox( "Select Explanation Length", 
                            [
                                "Short (1-2 paragraphs)",
                                "Medium (3-5 paragraphs)",
                                "Long (detailed explanation)"
                            ] 
                        )


template = load_prompt('template.json')


if st.button('Summarize'):
    chain = template | chat_model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)





