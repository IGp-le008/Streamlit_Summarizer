import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import OllamaLLM
from  langchain_core.prompts import load_prompt, PromptTemplate
import streamlit as st



# api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")


load_dotenv()
# model="meta-llama/Llama-3.1-8B"
model = OllamaLLM(model="llama3")

#for using online model 
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

chat_model = ChatHuggingFace(llm=llm)

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
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)





