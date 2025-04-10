from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal, Annotated
from copy import deepcopy
from openai.types.chat import ChatCompletionMessageParam
import chainlit as cl
import re
import os

class SearchSources(BaseModel):
    """Retrieve sources from the Azure AI Search index"""
    search_query: str = Field(..., description="Query string to retrieve documents from azure search eg: 'Health care plan'")

class AnswerReflectionWithStop(BaseModel):
    answer_reflection: str = Field(
        None, description="Explanation of why the previous answer was incorrect"
    )
    follow_up_action: Literal["index_search", "web_search", "stop"] = Field(
        ..., description="Follow up action to take to try to fix the answer."
    )
    follow_up_action_reason: str = Field(
        None, description="Explanation of why the follow up action was chosen"
    )
    query: Optional[str] = Field(
        None, description="Optional new search query suggested by the reflection"
    )

class StopTool(BaseModel):
    """
    There's no way to correct the previous answer.
    You only take this action if you've tried at least 1 action to correct the previous answer and it was unsuccessful.
    Take into account the previous actions taken to correct the previous answer.
    """

class WebSearchTool(BaseModel):
    """
    The Chat History and Sources are missing key information that is needed to correct the previous answer
    You need to search the web for additional public information.
    Suggest a new web search query to find the missing information.
    If the answer was incorrect or incomplete because the required information was not in the text sources, and the required information is public knowledge, you should suggest searching the web for additional information.
    Hint: If we previously made an index_search for a similar topic, it is unlikely the internal index has this information. If it seems like it might be public knowledge, you should suggest searching the web.
    """
    web_query: str = Field(..., description="Query string to retrieve documents from the web eg: 'Health care plan'")

class IndexSearchTool(BaseModel):
    """
    The Chat History and Sources are missing key information that is needed to correct the previous answer.
    You need to search an internal index for additional private information.
    Suggest a new search query to find the missing information.
    If the answer was incorrect or incomplete because the required information was not in the text sources, and the required information is internal to the organization, you should suggest searching the internal index for additional information.
    """
    index_query: str = Field(..., description="Query string to retrieve documents from azure search eg: 'Health care plan'")

class ToolChoiceReflection(BaseModel):
    answer_reflection: str = Field(
        ..., description="Explanation of why the previous answer was incorrect"
    )
    follow_up_action: Union[StopTool, WebSearchTool, IndexSearchTool] = Field(..., description="Follow up action to take to try to fix the answer.")
    follow_up_action_reason: str = Field(..., description="Reason we took the follow up action")

class EvaluatorResponse(BaseModel):
    type: str
    score: int
    explanation: str

class Document(BaseModel):
    content: str
    id: str
    score: float
    title: str

class ActionBase(BaseModel):
    type: str

    async def render(self):
        pass

    def step_name(self):
        return self.type

    def create_step(self):
        return True

    def render_during_step(self) -> bool:
        return True
    
    def get_input_props(self) -> Dict[str, Any]:
        return {}

    def get_output_props(self) -> Dict[str, Any]:
        return {}

class QuestionAction(ActionBase):
    type: Literal["question"] = "question"
    question_text: str

    def create_step(self):
        return False

    async def render(self):
        await cl.Message(content=self.question_text, type="user_message").send()

class QueryRewriteAction(ActionBase):
    type: Literal["query_rewrite"] = "query_rewrite"
    query: str
    rewritten_query: Optional[str] = None
    message_history: Optional[List[ChatCompletionMessageParam]] = None

    def get_input_props(self) -> Dict[str, Any]:
        return { "query": self.query, "message_history": self.message_history }

    def get_output_props(self) -> Dict[str, Any]:
        return { "rewritten_query": self.rewritten_query }

class FilterDocumentsAction(ActionBase):
    type: Literal["filter_documents"] = "filter_documents"
    threshold: float
    documents: Optional[List[Document]] = None
    filtered_documents: List[Document]

    def get_input_props(self) -> Dict[str, Any]:
        return { "threshold": self.threshold, "documents": self.documents }
    
    def get_output_props(self) -> Dict[str, Any]:
        return { "filtered_documents": self.filtered_documents }

class IndexSearchAction(ActionBase):
    type: Literal["index_search"] = "index_search"
    documents: List[Document]
    query: str
    rewritten_query: Optional[str] = None

    def get_input_props(self) -> Dict[str, Any]:
        return { "query": self.query, "rewritten_query": self.rewritten_query }

    def get_output_props(self) -> Dict[str, Any]:
        return { "documents": self.documents }

class WebCitation(BaseModel):
    name: str
    url: str
    title: str

class WebSearchAction(ActionBase):
    type: Literal["web_search"] = "web_search"
    query: str
    response_content: str
    citations: List[WebCitation]

class AnswerSources(BaseModel):
    message_history: List[dict]  # Assuming ChatCompletionMessageParam is a dict
    text_sources: Optional[List[Document]] = None
    public_sources: Optional[str] = None
    public_citations: Optional[List[WebCitation]] = None

    def create_context(self) -> str:
        context = "\n".join([message["role"] + ":\n" + message["content"] for message in self.message_history])
        if self.text_sources:
            context = "Sources\n"
            content_no_newlines = lambda document: document.content.replace("\n", " ").replace("\r", " ")
            context += '\n '.join([f'{document.title}: {content_no_newlines(document)}' for document in self.text_sources])
        if self.public_sources:
            context += "\nPublic Sources\n"
            public_sources = self.public_sources
            for details in self.public_citations:
                public_sources = public_sources.replace(details.name, f" [{details.title}]({details.url}) ")
            context += public_sources
        
        if len(context) == 0:
            context = "No sources found."
        return context

    def add_index_search(self, index_search_action: IndexSearchAction) -> "AnswerSources":
        text_sources = deepcopy(self.text_sources or []) 
        for document in index_search_action.documents:
            if not any(existing_document.id == document.id for existing_document in text_sources):
                text_sources.append(document)
        return AnswerSources(
            message_history=deepcopy(self.message_history),
            text_sources=text_sources,
            public_sources=self.public_sources,
            public_citations=self.public_citations
        )

    def add_web_search(self, web_search_action: WebSearchAction) -> "AnswerSources":
        public_citations = deepcopy(self.public_citations or [])
        for citation in web_search_action.citations:
            if not any(existing_citation.url == citation.url for existing_citation in public_citations):
                public_citations.append(citation)
        return AnswerSources(
            message_history=deepcopy(self.message_history),
            text_sources=self.text_sources,
            public_sources=(self.public_sources or "") + web_search_action.response_content,
            public_citations=public_citations
        )

    @staticmethod
    def from_index_search(message_history: Optional[List[ChatCompletionMessageParam]], index_search_action: IndexSearchAction) -> "AnswerSources":
        return AnswerSources(
            message_history=deepcopy(message_history or []),
            text_sources=index_search_action.documents
        )

    @staticmethod
    def from_web_search(message_history: Optional[List[ChatCompletionMessageParam]], web_search_action: WebSearchAction) -> "AnswerSources":
        return AnswerSources(
            message_history=deepcopy(message_history or []),
            public_sources=web_search_action.response_content,
            public_citations=web_search_action.citations
        )

class AnswerAction(ActionBase):
    type: Literal["answer"] = "answer"
    question_text: str
    answer_text: str
    answer_sources: AnswerSources

    async def render(self):
        await cl.Message(content=self.answer_text, elements=self.create_pdf_citations()).send()
    
    def step_name(self):
        return "generate_answer"
    
    def create_pdf_citations(self) -> List[cl.Pdf]:
        # Regex pattern to match citations like:
        # [AnyName.pdf#page=107]
        pattern = r'\[(?P<pdf>[^\]]+\.pdf)#page=(?P<page>\d+)\]'
        citation_names = set()
        citations = []
        for match in re.finditer(pattern, self.answer_text):
            pdf_name = match.group('pdf')
            page = int(match.group('page'))
            pdf_path = os.path.join("data", pdf_name)
            if os.path.exists(pdf_path) and pdf_path not in citation_names:
                citations.append(cl.Pdf(name=pdf_name, path=pdf_path, page=page))
                citation_names.add(pdf_path)
        
        return citations



    def render_during_step(self):
        return False
    
    def get_input_props(self) -> Dict[str, Any]:
        return { "question_text": self.question_text, "answer_sources": self.answer_sources.create_context() }

    def get_output_props(self) -> Dict[str, Any]:
        return { "answer_text": self.answer_text }

    def append_qa_message_history(self):
        self.answer_sources.message_history.append(
            {
                "role": "user",
                "content": self.question_text,
            }
        )
        self.answer_sources.message_history.append(
            {
                "role": "assistant",
                "content": self.answer_text,
            }
        )

class EvaluationAction(ActionBase):
    type: Literal["evaluation"] = "evaluation"
    evaluations: List[EvaluatorResponse]
    answer: AnswerAction

    def step_name(self):
        return "evaluate_answer"

    async def render(self):
        await cl.Message(content="", elements=[
            cl.CustomElement(name="EvaluationResults", props={"evaluations": [evaluation.model_dump() for evaluation in self.evaluations]}),
        ]).send()  # Send the evaluation message to the user

    def get_input_props(self) -> Dict[str, Any]:
        return { "answer": self.answer }

    def get_output_props(self):
        return { "evaluations": self.evaluations }

class ReflectionAction(ActionBase):
    type: Literal["reflection"] = "reflection"
    answer_text: str
    answer_reflection: str
    follow_up_action: Literal["index_search", "web_search", "stop"]
    follow_up_action_reason: str
    next_query: Optional[str] = None
    actions: List["Action"]

    def step_name(self):
        return "reflect_answer"

    async def render(self):
        await cl.Message(
            content="",
            elements=[
                cl.CustomElement(
                    name="ReflectionAction",
                    props={
                        "answer_reflection": self.answer_reflection,
                        "follow_up_action": self.follow_up_action,
                        "follow_up_action_reason": self.follow_up_action_reason,
                        "next_query": self.next_query
                    }
                )
            ]).send()  # Send the reflection message to the user

    def get_input_props(self) -> Dict[str, Any]:
        return { "answer_text": self.answer_text }

    def get_output_props(self) -> Dict[str, Any]:
        return { "answer_reflection": self.answer_reflection, "follow_up_action": self.follow_up_action, "follow_up_action_reason": self.follow_up_action_reason, "next_query": self.next_query }

class StopAction(ActionBase):
    type: Literal["stop"] = "stop"
    no_new_sources: bool = False

Action = Annotated[Union[QuestionAction, QueryRewriteAction, FilterDocumentsAction, IndexSearchAction, WebSearchAction, AnswerAction, EvaluationAction, ReflectionAction, StopAction], Field(discriminator='type')]

ReflectionAction.model_rebuild()

class ActionList(BaseModel):
    actions: List[Action]

    def get_last_answer(self) -> Optional[AnswerAction]:
        last_answer_action = next(
            (action for action in reversed(self.actions)
            if isinstance(action, AnswerAction)),
            None
        )
        return last_answer_action

    def get_last_query(self) -> Optional[str]:
        last_query_action = next(
            (action for action in reversed(self.actions)
            if isinstance(action, WebSearchAction) or isinstance(action, IndexSearchAction)),
            None
        )
        if last_query_action:
            if isinstance(last_query_action, WebSearchAction):
                return last_query_action.query
            elif isinstance(last_query_action, IndexSearchAction):
                return last_query_action.rewritten_query or last_query_action.query
        
        return None

    def get_last_evaluation(self) -> Optional[EvaluationAction]:
        last_evaluation_action = next(
            (action for action in reversed(self.actions)
            if isinstance(action, EvaluationAction)),
            None
        )
        return last_evaluation_action
