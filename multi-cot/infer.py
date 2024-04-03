# basics
import os
import sys
import json
import pandas as pd
import traceback
from io import StringIO
import joblib as jb

# pydantic
from typing import TypedDict, Annotated, Sequence
import operator

# self-defined functions
from util_functions import get_last_chains, save_new_chain, clean_table

# langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from datasets import load_dataset


# langgraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END


os.environ['OPENAI_API_KEY'] = "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"


# parser for action chain
def get_action(actions):
    if "<BEGIN>" in actions:
        a = actions.split('->')[1].strip()
    else:
        a = actions.split('->')[0].strip()
    return  a


# function to evaluate the next action in the chain
@tool
def evaluate_pandas_chain(
    chain: Annotated[str, "The pandas chain of actions . e.g. df1.groupby('age').mean() -> df1.sort_values() -> <END>"],
    inter=None):
    """Use this to execute the chain of pandas code on the dataframes"""

    name = "evaluate_pandas_chain"

    try:
        action = get_action(chain)#.replace(toreplace, 'inter')
        print('\n\naction: ', action)
        
        inter = eval(action, {"inter": inter, "df_dic": df_dic})
        
        if isinstance(inter, pd.DataFrame):
            intermediate = inter.head(50).to_markdown()
        else:
            intermediate = inter

        return intermediate, action, inter
                
    except Exception as e:
        return f"An exception occured: {traceback.format_exc()}", action, None

# function to look at dataframes 
@tool
def view_pandas_dataframes(
    df_list: Annotated[Sequence[str], "List of maximum 3 pandas dataframes you want to look at, e.g. [df1, df2, df3]"]):
    """Use this to view the head(10) of dataframes to answer your question"""

    name = "view_pandas_dataframes"

    markdown_str = "Here are .head(10) of the dataframes you requested to see:\n"
    for df in df_list:
        df_head = df_dic[df].head(10).to_markdown()
        markdown_str += f"{df}:\n{df_head}\n"

    markdown_str = markdown_str.strip()
    return markdown_str

tools = [evaluate_pandas_chain, view_pandas_dataframes]
tool_executor = ToolExecutor(tools)

functions = [convert_to_openai_function(t) for t in tools]

# we pull the prompt from langchain hub
SYSTEM_PROMPT = hub.pull("hrubyonrails/multi-cot").messages[0].prompt.template

# print the formatted prompt template?
print("System Prompt")
print("-"*50)
print(SYSTEM_PROMPT)
print("-"*50)


# create graph state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    actions: Annotated[Sequence[str], operator.add]
    inter: pd.DataFrame
    question: str
    memory: str


# Define the function that determines whether to continue or not: conditional edge
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue with the call_tool node
    else:
        return "continue"

# Define the function that calls the model
def call_model(state):
    
    response = model.invoke(state)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
def call_tool(state):
    messages = state['messages']
    
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    
    
    tool_input = last_message.additional_kwargs["function_call"]["arguments"]
    

    tool_input_dict = json.loads(tool_input)
    tool_input_dict['inter'] = state['inter']

    if last_message.additional_kwargs["function_call"]["name"] == 'view_pandas_dataframes':
        
        # We construct an ToolInvocation from the function_call
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=tool_input_dict,
        )
        # We call the tool_executor and get back a response
        response = tool_executor.invoke(action)

        function_message = FunctionMessage(content=str(response), name=action.tool)
        return {"messages": [function_message]} # ,"actions": [attempted_action]}
    

    # if the tool is to evaluate chain the chain
    elif last_message.additional_kwargs["function_call"]["name"] == 'evaluate_pandas_chain':
   
        # We construct an ToolInvocation from the function_call
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=tool_input_dict,
        )
        # We call the tool_executor and get back a response
        response, attempted_action, inter = tool_executor.invoke(action)
            
        if "An exception occured:" in str(response):
            error_info = f"""
            You have previously performed the actions: 
            {state['actions']}

            Current action: 
            {attempted_action}

            Result .head(50): 
            {response}

            You must correct your approach and continue until you can answer the question:
            {state['question']}

            Continue the chain with the following format: action_i -> action_i+1 ... -> <END>
            """
            print(error_info)

            function_message = FunctionMessage(content=str(error_info), name=action.tool)
            return {"messages": [function_message]}
        
        else:

            success_info = f"""
            You have previously performed the actions: 
            {state['actions']}

            Current action: 
            {attempted_action}

            Result .head(50):
            {response}

            You must continue until you can answer the question:
            {state['question']}

            Continue the  chain with the following format: action_i -> action_i+1 ... -> <END>
            """
            print(success_info)

            # We use the response to create a FunctionMessage
            function_message = FunctionMessage(content=str(success_info), name=action.tool)
            # We return a list, because this will get added to the existing list
            return {"messages": [function_message], "actions": [attempted_action], "inter": inter}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END
    }
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge('action', 'agent')

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()



ds = load_dataset("kpriyanshu256/MultiTabQA-atis")

P = []

for row in ds['test']:

    tables = row['tables']
    user_query = row['query']
    answer = row['answer']

    print("Query: ", user_query)

    df_list = [clean_table(table) if isinstance(table, pd.core.frame.DataFrame) else clean_table(
                pd.read_json(StringIO(table), orient='split')) for table in tables]

    answer_table = clean_table(answer) if isinstance(answer, pd.core.frame.DataFrame) else clean_table(
                pd.read_json(StringIO(answer), orient='split'))        

    questions_str = ""

    for i, x in enumerate(df_list):
        questions_str += f'df{i+1}: table with columns - {",".join(x.columns)}\n'

    print(questions_str)


    # create df_dic for use by python eval() in evaluate_pandas_chain
    df_dic = {}
    for i, dataframe in enumerate(df_list):
        df_dic[f"df{i + 1}"] = dataframe



    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    prompt = prompt.partial(num_dfs=len(df_list))
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    prompt = prompt.partial(questions_str=questions_str)

    # passing in past successful queries
    chain_examples = ""
    if(type(get_last_chains()) == pd.core.frame.DataFrame):
        for index, row in get_last_chains()[["query", "chain"]].iterrows():
            chain_examples += f'Question: {row["query"]}\nChain: {row["chain"]}\n\n'
    prompt = prompt.partial(chain_examples=chain_examples)


    # bind model
    model = prompt | ChatOpenAI(model="gpt-4-0125-preview").bind_functions(functions)



    inputs = {"messages": [HumanMessage(content=user_query)], "actions":["<BEGIN>"], "question": user_query, "memory": ""}

    for output in app.stream(inputs, {"recursion_limit": 40}):
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            if key == "agent":
                print("🤖 Agent working...")
            elif key == "action":
                if value["messages"][0].name == "view_pandas_dataframes":
                    print("🛠️ Current action:")
                    print("`viewing dataframes`")
                else:
                    if "actions" in value.keys():
                        print(f"🛠️ Current action:")
                        print(f"`{value['actions']}`")
                        print(f"Current output:")
                        print(value["inter"])
                    else:
                        print(f"⚠️ An error occured, retrying...")
            else:
                print("🏁 Finishing up...")
                print(f"Final output:")
                print(value["inter"])
                print(f"Final action chain:")
                print(" -> ".join(value["actions"])  + ' -> <END>')

            print("---")
            pass


    output_dict = output["__end__"]
    agent_response = output_dict["messages"][-1].content
    final_table = output_dict["inter"]
    final_message = agent_response.replace('<END>', '')

    print("Agent Message: ", final_message)
    print("Table: ", final_table)

    print("Truth: ", answer_table)

    P.append((answer_table, final_table))


jb.dump(P, "qa/multicot_atis.pkl")