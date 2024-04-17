import streamlit as st
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embeddings import download_gemini_embedding
from QAWithPDF.model_api import load_model

def main():
    st.set_page_config("QA with Documents")

    doc = st.file_uploader("upload your documents")

    st.header("QA with Documents(Information Returieval)")

    user_question= st.text_input("Ask your question")

    if st.button("submit & process"):
        with st.spinner("Precessing..."):
            documents=load_data/(doc)
            model=load_model()
            query_engine=download_gemini_embedding(model,documents)

            response = query_engine.query(user_question)

            st.write(response.response)


if __name__=="__main__":
    main() 
