import streamlit as st
import lancedb
#from openai import OpenAI
#from dotenv import load_dotenv
#from langchain import LLMChain, PromptTemplate
#from langchain.memory import ConversationBufferMemory
#from langchain.llms import Ollama
from langchain_ollama.llms import OllamaLLM
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
#from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain.callbacks.manager import CallbackManager
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
#load_dotenv()

# Initialize OpenAI client
#client = OpenAI()
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model = OllamaLLM(model="llama3.2", callbacks=callback_manager)
#callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#llm = Ollama(model=st.session_state.selected_model, callbacks=callback_manager)

# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        LanceDB table object
    """
    db = lancedb.connect("data/lancedb")
    return db.open_table("docling")


def get_context(query: str, table, num_results: int = 5) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    results = table.search(query).limit(num_results).to_pandas()
    contexts = []

    for _, row in results.iterrows():
        # Extract metadata
        filename = row["metadata"]["filename"]
        print("filename=" + filename)
        page_numbers = row["metadata"]["page_numbers"]
        print("page_numbers")
        print(page_numbers)
        title = row["metadata"]["title"]
        print("title =" + title)

        # Build source citation
        source_parts = []
        if filename:
            source_parts.append(filename)
        if page_numbers.any():
            source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

        source = f"\nSource: {' - '.join(source_parts)}"
        if title:
            source += f"\nTitle: {title}"

        contexts.append(f"{row['text']}{source}")

    return "\n\n".join(contexts)


def get_chat_response(messages, question: str, context: str) -> str:
    """Get streaming response from llama3.2 running locally.

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
    Use only the information from the context to answer questions. If you're unsure or the context
    doesn't contain the relevant information, say so.
    
    Question:
    {question}
    
    Context:
    {context}
    """

    sys_prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = sys_prompt | model
    ans = chain.invoke({"context": context, "question": question})
    #messages_with_context = [{"role": "system", "content": system_prompt}, *messages]

    # Create the streaming response
    # stream = client.chat.completions.create(
        #model="gpt-4o-mini",
        #messages=messages_with_context,
        #temperature=0.7,
        #stream=True,
    #)

    print(ans)
    # Use Streamlit's built-in streaming capability
    response = st.write(ans)
    return response


# Initialize Streamlit app
st.title("📚 Document Q&A")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize database connection
table = init_db()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context
    with st.status("Searching document...", expanded=False) as status:
        context = get_context(prompt, table)
        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.write("Found relevant sections:")
        for chunk in context.split("\n\n"):
            # Split into text and metadata parts
            parts = chunk.split("\n")
            text = parts[0]
            metadata = {
                line.split(": ")[0]: line.split(": ")[1]
                for line in parts[1:]
                if ": " in line
            }

            source = metadata.get("Source", "Unknown source")
            title = metadata.get("Title", "Untitled section")

            st.markdown(
                f"""
                <div class="search-result">
                    <details>
                        <summary>{source}</summary>
                        <div class="metadata">Section: {title}</div>
                        <div style="margin-top: 8px;">{text}</div>
                    </details>
                </div>
            """,
                unsafe_allow_html=True,
            )

    # Display assistant response first
    with st.chat_message("assistant"):
        # Get model response with streaming
        response = get_chat_response(st.session_state.messages, prompt, context)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
