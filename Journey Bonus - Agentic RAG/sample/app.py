import chainlit as cl

import os
from pathlib import Path

from agents import ActionList, AppState, index_search_only, reflection_single_step, reflection_multi_step
from azure.ai.evaluation import GroundednessEvaluator, RelevanceEvaluator
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from azure.search.documents.aio import SearchClient
from azure.ai.projects.aio import AIProjectClient

# Load environment variables
load_dotenv()

credential = AzureCliCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

evaluator_model_config = {
    "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],  # Replace with your Azure OpenAI endpoint
    "azure_deployment": os.environ["AZURE_OPENAI_EVALUATOR_DEPLOYMENT"],
    "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
    "api_key": os.environ["AZURE_OPENAI_API_KEY"],
}

app_state = AppState(
    openai_client = AsyncAzureOpenAI(
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider
    ),
    search_client = SearchClient(endpoint=os.environ["AZURE_SEARCH_ENDPOINT"], index_name=os.environ["AZURE_SEARCH_INDEX"], credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")) if os.getenv("AZURE_SEARCH_KEY") else credential),
    answer_generator_deployment = os.environ["AZURE_OPENAI_GENERATOR_DEPLOYMENT"],
    query_rewriter_deployment = os.environ["AZURE_OPENAI_REWRITER_DEPLOYMENT"],
    reflection_deployment = os.environ["AZURE_OPENAI_REFLECTION_DEPLOYMENT"],
    groundedness_evaluator = GroundednessEvaluator(evaluator_model_config),
    relevance_evaluator = RelevanceEvaluator(evaluator_model_config),
    ai_project_client = AIProjectClient.from_connection_string(os.environ["AI_PROJECT_CONN_STR"], credential),
    bing_grounding_name = os.environ["BING_GROUNDING_NAME"],
)

basic_chat_type = 'Basic'
single_reflection_type = 'Single Step Reflection'
multi_reflection_type = 'Multi Step Reflection'

commands = [
    {"id": basic_chat_type, "icon": "move-right", "description": "Use basic RAG strategy", "button": True},
    {"id": single_reflection_type, "icon": "corner-down-right", "description": "Use single step reflection RAG strategy", "button": True},
    {"id": multi_reflection_type, "icon": "iteration-ccw", "description": "Use multi step reflection RAG strategy", "button": True}
]

chat_type = multi_reflection_type

# Instrument the OpenAI client
cl.instrument_openai()

@cl.on_chat_start
async def start():
    await cl.context.emitter.set_commands(commands)
    await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="ChatType",
                label="Chat Type",
                values=[basic_chat_type, single_reflection_type, multi_reflection_type],
                initial_index=2
            ),
            cl.input_widget.Select(
                id="PreviousConversation",
                label="Load a previous conversation",
                values=[os.path.splitext(os.path.basename(k))[0] for k in Path("examples").glob("*.json")],
                initial_index=None
            )
        ]
    ).send()
    cl.user_session.set("actions", ActionList(actions=[]))

@cl.on_settings_update
async def update_settings(settings):
    global chat_type
    chat_type = settings["ChatType"]

    file = None
    if conversation := settings["PreviousConversation"]:
        file = f"examples/{conversation}.json"

    if file:
        with open(file, "r", encoding="utf-8") as f:
            action_list = ActionList.model_validate_json(f.read())
        
        for action in action_list.actions:
            if action.create_step():
                async with cl.Step(action.step_name()) as step:
                    step.input = action.get_input_props()
                    step.output = action.get_output_props()
                    await action.render()
            else:
                await action.render()
        
        session_action_list: ActionList = cl.user_session.get("actions")
        session_action_list.actions.extend(action_list.actions)

@cl.on_message
async def on_message(msg: cl.Message):
    question = msg.content

    message_chat_type = chat_type
    if msg.command:
        message_chat_type = msg.command
    
    session_action_list: ActionList = cl.user_session.get("actions")
    last_answer_action = session_action_list.get_last_answer()
    # If there was a last answer action, append its message history to the new question
    message_history = []
    if last_answer_action:
        last_answer_action.append_qa_message_history()
        message_history = last_answer_action.answer_sources.message_history

    if message_chat_type == basic_chat_type:
        action_list = await index_search_only(app_state, question, message_history)
    elif message_chat_type == single_reflection_type:
        action_list = await reflection_single_step(app_state, msg.content, message_history)
    elif message_chat_type == multi_reflection_type:
        action_list = await reflection_multi_step(app_state, msg.content, message_history)
    else:
        action_list = ActionList(actions=[])

    session_action_list.actions.extend(action_list.actions)

starters = [
 "What is included in my Northwind Health Plus plan that is not in standard?",
 "Who ensures the compliance of the Northwind Health Plus plan with state regulations?",
 "What is the deductible for prescription drugs provided by Northwind Health?"
]
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label=k,
            message=k,
        )
        for k in starters
    ]