from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM


### Document Loader
######################
def load_pdf(file_path):
    loader3 = PyPDFLoader(file_path) #your pdf file path
    papper = loader3.load_and_split()
    
    return papper


### Text Splitter
######################
def split_text(papper):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(papper)
    
    return all_splits


## Generating Embeddings with Ollama
######################
def generate_embeddings(all_splits):
    local_embeddings = OllamaEmbeddings(model="all-minilm")
    vectorstore = Chroma.from_documents(all_splits, local_embeddings)
    
    return vectorstore

### Implementing the Retrieval System
######################
def retrieve_documents(vectorstore, question):
    # Take the top 3 most relevant documents according to the question
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)
    context = ' '.join([doc.page_content for doc in retrieved_docs])
    
    return context

### Use LLM to create an answer using the selected part of the document and the question.
######################
def generate_answer(question, context):
    llm = OllamaLLM(model="gemma3")
    response = llm.invoke(f"""Answer the question according to the context given very briefly:
            Question: {question}.
            Context: {context}""")
    
    return response

def main():
    print("Hello! I am an assistant developed with the RAG method. You can ask questions related to your documents.")
        
    # Load the PDF document
    doc_path = input("Please enter the path to your PDF document: ")

    papper = load_pdf(doc_path)

    # Split the document into chunks
    all_splits = split_text(papper)

    ## Generating Embeddings with Ollama
    vectorstore = generate_embeddings(all_splits)

    while True:
        print("You can ask questions related to your documents. You can type 'exit' or 'q' to exit.")
        
        # Take user input for the question
        question = input("Please enter your question about the document: ")

        if question.lower() == 'exit' or question.lower() == 'q':
            print("Assistant is closing, By.")
            break

        
        # Retrieve relevant documents based on the question
        context = retrieve_documents(vectorstore, question)

        # Generate an answer using the LLM
        response = generate_answer(question, context)
    
        print("Answer:  ", response)
    return 

if __name__ == "__main__":
    main()