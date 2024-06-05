import os
import streamlit as st
import pickle
# import dill
import time
import langchain
import warnings
from pathlib import Path
from pathlib import Path
from langchain import OpenAI
from openai import OpenAI as openai_OpenAI
from IPython.display import Audio, display
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Ignore all warnings
warnings.filterwarnings("ignore")

# Ignore LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# Load OPENAI_API_KEY from dotenv file
from dotenv import load_dotenv
load_dotenv()

# Initialise LLM with required params
llm = OpenAI(temperature=0.9, max_tokens=500)

# Initialise client of OpenAI as openai_OpenAI from openai package
client = openai_OpenAI()

# Create the embeddings
embeddings = OpenAIEmbeddings()

st.title("Search Engine")
st.sidebar.title("URL Links")

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLS")

# create empty placeholder
main_placefolder = st.empty()

if process_url_clicked:

    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Loading Data .... Started... ")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
    main_placefolder.text("Splitting Data .... Started... ")

    # get document
    docs = text_splitter.split_documents(data)

    # Create the embeddings using openAIEmbeddings and save it to FAISS index
    vectorindex_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding Vector started .... ")
    time.sleep(2)

    # Storing vector index in local FAISS_store
    vectorindex_openai.save_local("FAISS_store")
##    vectorindex_loaded = FAISS.load_local("FAISS_store",embeddings, allow_dangerous_deserialization=True)

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists("FAISS_store"):

        # Load FAISS index from local
        vectorindex_loaded = FAISS.load_local("FAISS_store", embeddings, allow_dangerous_deserialization=True)

        # Create LangChain chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex_loaded.as_retriever())

        # Execute the chain with the query
        result = chain({"question": query}, return_only_outputs=True)

        # Display the result
        st.header("Answer")

##        st.subheader(result["answer"])
        st.write(result["answer"])

        speech_file_path = Path("C:\\PythonVM\\LLM\\SearchTool") / "speech1.mp3"
        response = client.audio.speech.create(
          model="tts-1",
          voice="alloy",
          input=result["answer"].strip(' \n')
        )
        response.stream_to_file(speech_file_path)
        # st.audio(Audio(speech_file_path, autoplay=True))
        # st.audio(speech_file_path, format="audio/mpeg")

        # Read the file as bytes
        with open(speech_file_path, "rb") as file:
            mp3_data = file.read()

        # mp3_url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
        st.audio(mp3_data, format="audio/mpeg")

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            source_list = sources.split("\n") # split the sources ny newlline
            for source in source_list:
                st.write(source)
    else:
        st.error("FAISS_store folder does not exist. Please process URLs first.")
