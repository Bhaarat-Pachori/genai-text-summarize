import os
from dotenv import load_dotenv

import openai
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o")
loader = PyPDFLoader("apjspeech.pdf")
docs = loader.load()


final_documents = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                 chunk_overlap=100).split_documents(docs)

chunks_prompt="""
                    Please summarize the below speech:
                    Speech:`{text}'
                    Summary:
            """

map_prompt_template = PromptTemplate(input_variables=['text'],
                                    template=chunks_prompt)

final_prompt = """
                    Provide the final summary of the entire speech with these important points.
                    Speech:{text}
                """

final_prompt_template = PromptTemplate(input_variables=['text'],
                                       template=final_prompt)

final_summary_chain = load_summarize_chain(llm=llm,
                             chain_type="map_reduce",
                             map_prompt=map_prompt_template,
                             combine_prompt=final_prompt_template,
                             verbose=True,
                             )

final_summary_chain.run(final_documents)
