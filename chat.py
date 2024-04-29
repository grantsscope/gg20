import streamlit as st
import os

from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder

from langchain.agents.agent_types import AgentType
from langchain.tools import Tool, tool
from trubrics.integrations.streamlit import FeedbackCollector

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

collector = FeedbackCollector(
    project="GG20",
    email=st.secrets["TRUBRICS_EMAIL"],
    password=st.secrets["TRUBRICS_PWD"],
)

st.set_page_config(
    page_title="GrantsScope",
    page_icon="🔎",
    layout="wide",
    #initial_sidebar_state="expanded",
)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

st.sidebar.markdown("## Important Links:")
st.sidebar.markdown("- Use [Explorer](https://explorer.gitcoin.co/#/) to search rounds and projects")
st.sidebar.markdown("- GG20 Donation and Leaderboard [Dashboard](https://www.gitcoin.co/grants-data)")
st.sidebar.markdown("- [Gitcoin Grants Portal for Donors](https://grants-portal.gitcoin.co/gitcoin-grants-donor-portal) for how-to videos and resources")
st.sidebar.markdown("- About [GrantsScope](http://grantsscope.xyz/)")

st.title('GrantsScope - GG20')
st.markdown('Ask away your questions to learn more about the grantees in GG20. Here are a few sample questions to start your discovery:')
st.markdown('*- Give me examples of real-world impact for grantees in climate round.*')
st.markdown('*- I am interested in knowing about grantees working in <add a category you are interested in>.*')
st.markdown('*- ELI5 the project <project name> to me.*')
st.markdown('*- Why should I donate to <add a project name>?*')
#st.info('Development status: Ready for Climate Round. Other rounds coming soon!')
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.link_button("Support GrantsScope in GG20", "https://explorer.gitcoin.co/#/round/42161/25/149",type="primary")
    
@st.cache_resource(ttl="1h")

def configure_retriever_grantees():
    index = './storage/csv_index'
    embeddings = OpenAIEmbeddings()    
    vectorstore = FAISS.load_local(index, embeddings,allow_dangerous_deserialization= True )
    return vectorstore.as_retriever()

def configure_retriever_rounds():
    index = './storage/txt_index'
    embeddings = OpenAIEmbeddings()    
    vectorstore = FAISS.load_local(index, embeddings,allow_dangerous_deserialization= True )
    return vectorstore.as_retriever()


grantee_info = create_retriever_tool(
    configure_retriever_grantees(),
    "Grantee_Discovery",
    "Use this tool to answer questions related to projects and grantees based on round they are participating in. If the answer is not available in the context information, respond as unable to find an answer."
)

round_info = create_retriever_tool(
    configure_retriever_rounds(),
    "GG20_Rounds",
    "Use this tool to answer questions related to GG20 and rounds.  If the answer is not available in the context information, respond as unable to find an answer."
)

#tools = [grantee_info, round_info]
tools = [grantee_info]

llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
#memory = AgentTokenBufferMemory(llm=llm)

message = SystemMessage(
    content=(
        "If the answer is not available in the context information, respond as unable to find an answer. \
        When sharing information about a project, share which round they are part of (use round name and not round number), the website and the Explorer Link."
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt, )

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    #return_intermediate_steps=True,
)

starter_message =   "Ask me anything about the grantees in GG20 Rounds!"
                    

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]

if "logged_prompt" not in st.session_state:
    st.session_state.logged_prompt = None

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    #memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("assistant"):

        latest_messages = ""
        # Capture the latest two responses as additional context to the prompt
        if 'messages' in st.session_state and len(st.session_state.messages) >= 4:
            latest_messages = st.session_state.messages[-4:]
        else:
            latest_messages = st.session_state.messages

        # Debug
        #st.markdown("Additional context includes **** ")
        #for msg in latest_messages:
        #    st.markdown(msg)    
        #st.markdown("***")

        #st_callback = StreamlitCallbackHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        
        try:
            response = agent_executor(
                #{"input": prompt},
                {"input": prompt, "history": latest_messages},
                #callbacks=[st_callback],
                callbacks=[stream_handler],
                include_run_info=True,
            )

            st.session_state.messages.append(AIMessage(content=response["output"]))
            #st.markdown(response["output"])
            #memory.save_context({"input": prompt}, response)
            #st.session_state["messages"] = memory.buffer
            run_id = response["__run"].run_id

            st.session_state.logged_prompt = collector.log_prompt(
                config_model={"model": "gpt-4-turbo"},
                prompt=prompt,
                generation=response["output"],
            )


        except:
            st.markdown("The dude who made me doesn't have access to models with longer context yet, or, in English, my brain exploded trying to compress all the information needed to answer your question.")
            st.markdown("Please refresh the browser and try asking this a little differently. I will try to remain sane!")
            st.markdown("![Exploding brain meme](https://media.tenor.com/InOgyW0EIEcAAAAC/exploding-brain-mind-blown.gif)")

if st.session_state.logged_prompt:
    user_feedback = collector.st_feedback(
        component="feedback",
        feedback_type="thumbs",
        open_feedback_label="[Optional] Provide additional feedback",
        model="gpt-4-turbo",
        prompt_id=st.session_state.logged_prompt.id,
    )
