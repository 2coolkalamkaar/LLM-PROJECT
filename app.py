from fastapi import FastAPI, Form, Request, Response, File
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain.llms import HuggingFacePipeline
from langchain.chains import QAGenerationChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import os
import json
import csv
import aiofiles
from dotenv import load_dotenv
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Set Hugging Face API token
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def load_llm():
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

def file_processing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''
    for page in data:
        question_gen += page.page_content

    splitter_ques_gen = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen

def llm_pipeline(file_path):
    try:
        document_ques_gen, document_answer_gen = file_processing(file_path)
        llm_ques_gen_pipeline = load_llm()
        print("Question Generation Model Loaded")

        if not document_ques_gen or not document_answer_gen:
            print("Error: No documents found for question or answer generation.")
            return None, []

        question_text = "\n".join([doc.page_content for doc in document_ques_gen if isinstance(doc.page_content, str)])
        print("Input Text for Questions:", question_text)

        if not question_text.strip():
            print("Error: Input text is empty after processing.")
            return None, []

        prompt_template = """
        You are an expert at creating questions based on the provided text.
        Your goal is to prepare questions on the topic "Indian Constitution" using data from the given text file.
        You do this by asking questions about the text below:

        ------------
        {text}
        ------------

        Prepare a set of 4 questions and answers on "Indian Constitution" using data from the given text file.
        QUESTIONS:
        """
        PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

        ques_gen_chain = QAGenerationChain.from_llm(
            llm=HuggingFacePipeline(pipeline=llm_ques_gen_pipeline),
            prompt=PROMPT_QUESTIONS
        )

        ques = ques_gen_chain.run({"text": question_text})
        print("Generated Questions:", ques)

        if not ques:
            print("Error: No valid questions generated.")
            return None, []

        embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = FAISS.from_documents(document_answer_gen, embeddings)

        llm_answer_gen = load_llm()
        filtered_ques_list = [q for q in ques.split("\n") if q.endswith('?') or q.endswith('.')]

        answer_generation_chain = RetrievalQA.from_chain_type(
            llm=HuggingFacePipeline(pipeline=llm_answer_gen),
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )

        return answer_generation_chain, filtered_ques_list

    except Exception as e:
        print(f"Error in llm_pipeline: {e}")
        return None, []

def get_csv(file_path):
    answer_generation_chain, ques_list = llm_pipeline(file_path)

    if answer_generation_chain is None or not ques_list:
        print("Error: No valid answer generation chain or questions generated.")
        return None

    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)

    output_file = os.path.join(base_folder, "QA.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])

        for question in ques_list:
            print("Question: ", question)
            try:
                answer = answer_generation_chain({"query": question})
                print("Answer: ", answer)
                csv_writer.writerow([question, answer['result']])
            except Exception as e:
                print(f"Error processing question: {question}\n{e}")
                csv_writer.writerow([question, "Error generating answer"])

    return output_file

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    base_folder = 'static/docs/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    pdf_filename = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_filename, 'wb') as f:
        await f.write(pdf_file)

    response_data = jsonable_encoder({"msg": 'success', "pdf_filename": pdf_filename})
    return Response(content=json.dumps(response_data), media_type="application/json")

@app.post("/analyze")
async def analyze(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)
    if output_file is None:
        return Response(content=json.dumps({"error": "Failed to generate questions and answers."}), media_type="application/json")

    response_data = jsonable_encoder({"output_file": output_file})
    return Response(content=json.dumps(response_data), media_type="application/json")

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)
