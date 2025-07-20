# %% Imports
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_ollama import ChatOllama

# %% LangFuse Setup
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

langfuse = Langfuse(
    public_key="pk-lf-c77d9846-34ed-4d02-943e-e2ecea5e7890",
    secret_key="sk-lf-562eec98-f294-4182-b473-ae2c1d7c2db2",
    host="http://localhost:3000",
)

langfuse_handler = CallbackHandler()


# %% Get Session id
def get_session_id() -> str:
    return str(uuid4())


# %% Configuration
def get_config(not_all_unique=False) -> dict:
    session_id = get_session_id()
    return {
        "callbacks": [langfuse_handler],
        "metadata": {"langfuse_session_id": session_id},
    }


# %%
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Answer all questions to the best
        of your ability.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# %%
model = ChatOllama(model="gemma3:1b")

# %%
chain = prompt | model

# %%
chain.invoke(
    {
        "messages": [
            (
                "human",
                """Translate this sentence from English to French: I love
            programming.""",
            ),
            ("ai", "J'adore programmer."),
            ("human", "What did you just say?"),
        ],
    },
    config=get_config(),
)

# %%
chain


# %% Defining types
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages`
    # function in the annotation defines how this state should
    # be updated (in this case, it appends new messages to the
    # list, rather than replacing the previous messages)
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

# %%

model = ChatOllama(model="gemma3:1b")


def chatbot(state: State):
    answer = model.invoke(state["messages"], config=get_config())
    return {"messages": [answer]}


# The first argument is the unique node name
# The second argument is the function or Runnable to run
builder.add_node("chatbot", chatbot)

# %%
from langgraph.graph import END, START

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()
# %%
graph

# %%
from langchain_core.messages import HumanMessage

input = {"messages": [HumanMessage("hi!")]}
for chunk in graph.stream(input):
    print(chunk)
# %%
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())

# %%
thread1 = {"configurable": {"thread_id": "1"}}
result_1 = graph.invoke({"messages": [HumanMessage("hi, my name is Jack!")]}, thread1)

# %%
result_2 = graph.invoke({"messages": [HumanMessage("what is my name?")]}, thread1)

# %%
graph.get_state(thread1)

# %%
graph.update_state(thread1, [HumanMessage("I like LLMs!")])

# %%
from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langchain_ollama import ChatOllama

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=ChatOllama(model="gemma3:1b"),
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="what's 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages, config=get_config())

# %%
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    filter_messages,
)

messages = [
    SystemMessage("you are a good assistant", id="1"),
    HumanMessage("example input", id="2", name="example_user"),
    AIMessage("example output", id="3", name="example_assistant"),
    HumanMessage("real input", id="4", name="bob"),
    AIMessage("real output", id="5", name="alice"),
]

filter_messages(messages, include_types="human")

# %%
filter_messages(messages, exclude_names=["example_user", "example_assistant"])

"""
[SystemMessage(content='you are a good assistant', id='1'),
HumanMessage(content='real input', name='bob', id='4'),
AIMessage(content='real output', name='alice', id='5')]
"""

filter_messages(messages, include_types=[HumanMessage, AIMessage], exclude_ids=["3"])

"""
[HumanMessage(content='example input', name='example_user', id='2'),
 HumanMessage(content='real input', name='bob', id='4'),
 AIMessage(content='real output', name='alice', id='5')]
"""

# %%
model = ChatOllama(model="gemma3:1b")

filter_ = filter_messages(exclude_names=["example_user", "example_assistant"])

chain = filter_ | model

# %%
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)

messages = [
    SystemMessage("you're a good assistant."),
    SystemMessage("you always respond with a joke."),
    HumanMessage([{"type": "text", "text": "i wonder why it's called langchain"}]),
    HumanMessage("and who is harrison chasing anyway"),
    AIMessage(
        """Well, I guess they thought "WordRope" and "SentenceString" just 
        didn\'t have the same ring to it!"""
    ),
    AIMessage("""Why, he's probably chasing after the last cup of coffee in the 
        office!"""),
]

merge_message_runs(messages)

# %%
model = ChatOllama(model="gemma3:1b")
merger = merge_message_runs()
chain = merger | model

# %%
from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

model = ChatOllama(model="gemma3:1b")


class State(TypedDict):
    # Messages have the type "list". The `add_messages`
    # function in the annotation defines how this state should
    # be updated (in this case, it appends new messages to the
    # list, rather than replacing the previous messages)
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# %%
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# useful to generate SQL query
model_low_temp = ChatOllama(model="gemma3:1b", temperature=0.1)
# useful to generate natural language outputs
model_high_temp = ChatOllama(model="gemma3:1b", temperature=0.7)


class State(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input
    user_query: str
    # output
    sql_query: str
    sql_explanation: str


class Input(TypedDict):
    user_query: str


class Output(TypedDict):
    sql_query: str
    sql_explanation: str


generate_prompt = SystemMessage(
    """You are a helpful data analyst who generates SQL queries for users based 
    on their questions. Database System is PostGreSQL and database name is students and its associated table are in records."""
)


def generate_sql(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    messages = [generate_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(messages, config=get_config(not_all_unique=True))
    return {
        "sql_query": res.content,
        # update conversation history
        "messages": [user_message, res],
    }


explain_prompt = SystemMessage(
    "You are a helpful data analyst who explains SQL queries to users."
)


def explain_sql(state: State) -> State:
    messages = [
        explain_prompt,
        # contains user's query and SQL query from prev step
        *state["messages"],
    ]
    res = model_high_temp.invoke(messages, config=get_config(not_all_unique=True))
    return {
        "sql_explanation": res.content,
        # update conversation history
        "messages": res,
    }


builder = StateGraph(State, input=Input, output=Output)
builder.add_node("generate_sql", generate_sql)
builder.add_node("explain_sql", explain_sql)
builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "explain_sql")
builder.add_edge("explain_sql", END)

graph = builder.compile()
# %%
graph
# %%
graph.invoke({"user_query": "What is the total sales for each product?"})

# %%
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

embeddings = OllamaEmbeddings(model="nomic-embed-text")
# useful to generate SQL query
model_low_temp = ChatOllama(model="gemma3:1b", temperature=0.1)
# useful to generate natural language outputs
model_high_temp = ChatOllama(model="gemma3:1b", temperature=0.7)


class State(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input
    user_query: str
    # output
    domain: Literal["records", "insurance"]
    documents: list[Document]
    answer: str


class Input(TypedDict):
    user_query: str


class Output(TypedDict):
    documents: list[Document]
    answer: str


medical_records_store = InMemoryVectorStore.from_documents([], embeddings)
medical_records_retriever = medical_records_store.as_retriever()

insurance_faqs_store = InMemoryVectorStore.from_documents([], embeddings)
insurance_faqs_retriever = insurance_faqs_store.as_retriever()

router_prompt = SystemMessage(
    """You need to decide which domain to route the user query to. You have two 
        domains to choose from:
          - records: contains medical records of the patient, such as 
          diagnosis, treatment, and prescriptions.
          - insurance: contains frequently asked questions about insurance 
          policies, claims, and coverage.

Output only the domain name."""
)

session_id = str(uuid4())


def router_node(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    messages = [router_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(
        messages,
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )
    return {
        "domain": res.content,
        # update conversation history
        "messages": [user_message, res],
    }


def pick_retriever(
    state: State,
) -> Literal["retrieve_medical_records", "retrieve_insurance_faqs"]:
    if state["domain"] == "records":
        return "retrieve_medical_records"
    else:
        return "retrieve_insurance_faqs"


def retrieve_medical_records(state: State) -> State:
    documents = medical_records_retriever.invoke(
        state["user_query"],
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )
    return {
        "documents": documents,
    }


def retrieve_insurance_faqs(state: State) -> State:
    documents = insurance_faqs_retriever.invoke(
        state["user_query"],
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )
    return {
        "documents": documents,
    }


medical_records_prompt = SystemMessage(
    """You are a helpful medical chatbot who answers questions based on the 
        patient's medical records, such as diagnosis, treatment, and 
        prescriptions."""
)

insurance_faqs_prompt = SystemMessage(
    """You are a helpful medical insurance chatbot who answers frequently asked 
        questions about insurance policies, claims, and coverage."""
)


def generate_answer(state: State) -> State:
    if state["domain"] == "records":
        prompt = medical_records_prompt
    else:
        prompt = insurance_faqs_prompt
    messages = [
        prompt,
        *state["messages"],
        HumanMessage(f"Documents: {state['documents']}"),
    ]
    res = model_high_temp.invoke(
        messages,
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )
    return {
        "answer": res.content,
        # update conversation history
        "messages": res,
    }


builder = StateGraph(State, input=Input, output=Output)
builder.add_node("router", router_node)
builder.add_node("retrieve_medical_records", retrieve_medical_records)
builder.add_node("retrieve_insurance_faqs", retrieve_insurance_faqs)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START, "router")
builder.add_conditional_edges("router", pick_retriever)
builder.add_edge("retrieve_medical_records", "generate_answer")
builder.add_edge("retrieve_insurance_faqs", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()

# %%
graph
# %%
input = {"user_query": "Am I covered for COVID-19 treatment?"}
for c in graph.stream(input):
    print(c)
# %%
