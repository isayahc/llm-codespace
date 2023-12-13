from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp


# MODEL_ID = "TheBloke/zephyr-7B-beta-GGUF"
MODEL_ID = "TheBloke/Merged-DPO-7B-GGUF"
MODEL_BASENAME = "merged-dpo-7b.Q2_K.gguf"

CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = 1024

model_path = hf_hub_download(
  repo_id=MODEL_ID,
  filename=MODEL_BASENAME,
  resume_download=True,
  cache_dir="./models",
)

kwargs = {
  "model_path": model_path,
  "n_ctx": CONTEXT_WINDOW_SIZE,
  "max_tokens": MAX_NEW_TOKENS,
  "n_gpu_layers":4,
}

llm = LlamaCpp(
  model_path=model_path,
  temperature=0.1,
  n_ctx=4096,
  max_tokens=1024,
  n_batch=100,
  top_p=1,
  verbose=True,
  n_gpu_layers=100,
  )

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