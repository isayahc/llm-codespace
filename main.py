from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from src.llms.merged_dpo_llm import dpo_llm as llm


question = "How to turn cannabis flower into a tincure "

template = """Question: {question}


Answer: Let's think step by step."""

prompt = PromptTemplate(
    template=template,
    input_variables=[
        "question",
        ]
        )

llm_chain = LLMChain(prompt=prompt, llm=llm)

sample = llm_chain({"question":question,})

print(sample)
x=0 # for breakpoint