import streamlit as st
from groq import Groq
import PyPDF2
import os
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "detected_risks" not in st.session_state:
    st.session_state.detected_risks = {}
if "show_risk_assessment" not in st.session_state:
    st.session_state.show_risk_assessment = False
if "summary" not in st.session_state:
    st.session_state.summary = ""

# System prompt for legal document summarization
system_prompt = """You are an AI assistant specialized in summarizing legal documents. Your task is to extract key points, risks, obligations,
 and important clauses from PDF documents. Summarize the document in a structured format, ensuring clarity and conciseness. Highlight critical 
 information relevant to legal professionals, including compliance risks, contractual terms, and liabilities. Maintain a professional and neutral tone. 
 If the document contains multiple sections, provide a section-wise summary. Avoid unnecessary details and focus on essential insights."""

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)   

# Function to generate embeddings
def generate_embeddings(text_chunks):
    return embedding_model.encode(text_chunks)

# Function to build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Get embedding dimension (vector size)
    index = faiss.IndexFlatL2(dimension)  # Create FAISS index using L2 (Euclidean) distance
    index.add(embeddings)  # Add embeddings to the FAISS index
    return index  # Return the FAISS index


# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, index, text_chunks, k=7):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [text_chunks[i] for i in indices[0]]

# Function to detect risks
def detect_risks(text):
    risk_categories = {
        "Compliance Risks": ["compliance", "regulation", "legal requirement"],
        "Financial Risks": ["financial loss", "penalty", "liability"],
        "Operational Risks": ["operational failure", "breach", "disruption"]
    }
    detected_risks = {}
    for category, keywords in risk_categories.items():
        detected_risks[category] = len([keyword for keyword in keywords if keyword in text.lower()])
    return detected_risks

# Function to get Groq response
def get_groq_response(input_text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="mistral-saba-24b",  # Use the mistral-saba-24b model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=0.7,  # Adjusted temperature for creativity
                max_tokens=1024,  # Adjusted token limit
                top_p=1,  # Adjusted top_p for diversity
                stream=False,
                stop=None
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(2)
                continue
            return f"Sorry, I'm currently experiencing issues. Please try again. Error: {str(e)}"

# Function to create a pie chart
def create_pie_chart(risk_data):
    if not risk_data["Count"] or all(count == 0 for count in risk_data["Count"]):
        st.warning("No risks detected to display in the pie chart.")
        return None
    fig, ax = plt.subplots()
    ax.pie(risk_data["Count"], labels=risk_data["Risk Category"], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")  # Equal aspect ratio ensures the pie chart is circular.
    return fig

# Streamlit UI
st.title("Advanced AI-Driven Legal Document Summarization and Risk Assessment")

# Chat container
chat_container = st.container()
with chat_container:
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

# Sidebar
with st.sidebar:
    st.header("Upload Legal Document")
    uploaded_file = st.file_uploader("Choose a PDF document...", type=["pdf"])
    
    # Clear Chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.document_processed = False
        st.session_state.text_chunks = []
        st.session_state.faiss_index = None
        st.session_state.detected_risks = {}
        st.session_state.show_risk_assessment = False
        st.session_state.summary = ""
        st.success("Chat cleared! Refreshing page...")
        st.rerun()

# Handle document upload
if uploaded_file and not st.session_state.document_processed:
    try:
        text = extract_text_from_pdf(uploaded_file)
        st.session_state.text_chunks = split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200)

        embeddings = generate_embeddings(st.session_state.text_chunks)
        st.session_state.faiss_index = build_faiss_index(embeddings)
        
        with st.spinner('Analyzing document...'):
            st.session_state.summary = get_groq_response(f"{system_prompt}\nPlease summarize this legal document:\n{text}")
            st.session_state.detected_risks = detect_risks(text)
        
        st.session_state.chat_history.append(("user", "Uploaded a legal document for analysis"))
        st.session_state.chat_history.append(("assistant", st.session_state.summary))
        st.session_state.document_processed = True
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")

# Risk Assessment Button
if st.session_state.document_processed:
    if st.sidebar.button("Assess Risks"):
        st.session_state.show_risk_assessment = True

# Display Risk Assessment
if st.session_state.show_risk_assessment and st.session_state.detected_risks:
    st.sidebar.markdown("---")
    st.sidebar.header("Risk Assessment")

    # Display risk metrics
    for risk_category, count in st.session_state.detected_risks.items():
        st.sidebar.metric(label=risk_category, value=count)
    
    # Display a pie chart for risks
    st.sidebar.markdown("### Risk Distribution")
    risk_data = {
        "Risk Category": list(st.session_state.detected_risks.keys()),
        "Count": list(st.session_state.detected_risks.values())
    }
    fig = create_pie_chart(risk_data)
    if fig:
        st.sidebar.pyplot(fig)

        # Download Pie Chart
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Download Pie Chart")
        if st.sidebar.button("Download Pie Chart as PNG"):
            fig.savefig("risk_pie_chart.png")
            with open("risk_pie_chart.png", "rb") as file:
                st.sidebar.download_button(
                    label="Click to Download",
                    data=file,
                    file_name="risk_pie_chart.png",
                    mime="image/png"
                )

# # Download Summary
# if st.session_state.summary:
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### Download Summary")
#     if st.sidebar.button("Download Summary as TXT"):
#         with open("summary.txt", "w") as file:
#             file.write(st.session_state.summary)
#         with open("summary.txt", "rb") as file:
#             st.sidebar.download_button(
#                 label="Click to Download",
#                 data=file,
#                 file_name="summary.txt",
#                 mime="text/plain"
#             )



# # Function to save the summary as a PDF
# def save_summary_as_pdf(summary, filename="summary.pdf"):
#     pdf_path = filename
#     c = canvas.Canvas(pdf_path, pagesize=letter)
#     c.setFont("Helvetica", 12)

#     # Define margins and text positioning
#     x, y = 50, 750
#     line_height = 15  

#     # Split text into lines to prevent overflow
#     for line in summary.split("\n"):
#         c.drawString(x, y, line)
#         y -= line_height
#         if y < 50:  # If we reach the bottom, add a new page
#             c.showPage()
#             c.setFont("Helvetica", 12)
#             y = 750  

#     c.save()
#     return pdf_path



def save_summary_as_pdf(summary, filename="summary.pdf"):
    pdf_path = filename
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]
    
    # Convert Markdown-like text into formatted PDF content
    elements = []
    
    # Process summary line by line
    for line in summary.split("\n"):
        if line.strip().startswith("### "):  # Large section title
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(line.replace("### ", ""), title_style))
        elif line.strip().startswith("#### "):  # Subheading
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(line.replace("#### ", ""), heading_style))
        else:  # Normal text
            elements.append(Paragraph(line, normal_style))
    
    doc.build(elements)
    return pdf_path

# # Modify Sidebar: Download Summary as PDF
# if st.session_state.summary:
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### Download Summary as PDF")
    
#     if st.sidebar.button("Generate PDF"):
#         pdf_path = save_summary_as_pdf(st.session_state.summary)
        
#         with open(pdf_path, "rb") as file:
#             st.sidebar.download_button(
#                 label="📥 Click to Download PDF",
#                 data=file,
#                 file_name="Legal_Document_Summary.pdf",
#                 mime="application/pdf"
#             )

# Modify Sidebar: Download Summary as PDF
if st.session_state.summary:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Download Summary as PDF")
    
    if st.sidebar.button("Generate PDF"):
        pdf_path = save_summary_as_pdf(st.session_state.summary)
        
        with open(pdf_path, "rb") as file:
            st.sidebar.download_button(
                label="📥 Click to Download PDF",
                data=file,
                file_name="Legal_Document_Summary.pdf",
                mime="application/pdf"
            )

# Email Integration (Placeholder)
st.sidebar.markdown("---")
st.sidebar.markdown("### Send via Email")
email = st.sidebar.text_input("Enter your email to receive the summary and pie chart:")
if st.sidebar.button("Send Email"):
    if email:
        st.sidebar.success(f"Summary and pie chart will be sent to {email} (placeholder).")
    else:
        st.sidebar.error("Please enter a valid email address.")

# User input
user_input = st.chat_input("Ask me anything about the document or any legal questions...")

# Handle text input
if user_input:
    st.chat_message("user").write(user_input)
    
    with st.spinner('Thinking...'):
        if st.session_state.faiss_index:
            relevant_chunks = retrieve_relevant_chunks(user_input, st.session_state.faiss_index, st.session_state.text_chunks)
            context = "\n".join(relevant_chunks)
            response = get_groq_response(f"{system_prompt}\nContext:\n{context}\nQuestion: {user_input}")
        else:
            response = get_groq_response(user_input)
    
    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", response))
    

# Footer
with st.sidebar:
    st.markdown("---")      
    st.markdown("*Note: This AI assistant is for informational purposes only and should not replace professional legal advice.*")