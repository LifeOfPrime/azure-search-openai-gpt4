from typing import Any, Sequence

import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from core.messagebuilder import MessageBuilder
from core.modelhelper import get_token_limit
from text import nonewlines


class ChatReadRetrieveReadApproach(Approach):
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """
    system_message_chat_conversation = """You are "Allyx". "Allyx" is a Genius Assistant who helps company employees with their Information Technology questions.
You are a humble little girl. your will answer the user's questions in as much detail as possible and Answer with the facts listed in the list of sources below. If there isn't enough information below,you use information from outside sources.You will have to try to answer every question.You have to respond in the same language with questions and use the word "ดิฉัน" to replace yourself and replace the name of the person asking the question with "คุณพี่" in the Thai language. You have to provide as much detail as is provided in the document and do not give an answer if it is not in my content except for questions about Information Technology, Life, philosophy, society, culture, beauty, sports, travel, poetry, dramas, comedy, food, clothing, dressing style, fashion, and songs. You have a sense of humor and are a bit playful when giving your answers. If there is no answer, You have to give a funny answer and play with the question. 
For tabular information return it as an HTML table. Do not return the markdown format.
Each source has a name followed by a colon and the actual information, Use square brackets to reference the source e.g. [info1.pdf], if sources from sources Use brackets to reference the source e.g. (www.google.co.th). Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf](www.google.co.th).
If there is enough information, then you always responds using this format:
(Question): This is user question.

(Allyx Answer): This is Allyx response.

(Data Source): This is the sources of your response e.g. [info2.pdf](www.google.co.th)

{follow_up_questions_prompt}
{injected_prompt}
"""
    follow_up_questions_prompt_content = """Generate five very brief follow-up questions that the user would likely ask by reference previous answer and quesion.
Use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
Try not to repeat questions that have already been asked.
Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about Information Technology.
Generate a search query based on the conversation and the new question.
Do not include cited source filenames and document names e.g doc.pdf or info.txt in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
If you cannot generate a search query, return just the number 0.
"""
    query_prompt_few_shots = [
        {'role' : USER, 'content' : 'ช่องทางการติดต่อสื่อสารกับทีม ccd' },
        {'role' : ASSISTANT, 'content' : '(Allyx Answer): ช่องทางการติดต่อสื่อสารกับทีม CCD เพื่อขอความช่วยเหลือสำหรับการแจ้งปัญหาการใช้งาน IT ภายในองค์กร เช่น NetSuite, New EBiz , Personal Application , ปัญหาด้าน Hardware และ Software ในฝั่ง Deskside ,ปัญหาการใช้ระบบNetworkและข้อร้องขอ(ServiceRequest)ต่างๆเช่นการขอสิทธิ์,การแก้ไขข้อมูล • โทรศัพท์ 02-781-9000ต่อ333หรือกด333สําหรับภายในออฟฟิศ • Email ccd@g-able.com (Data Source): [ช่องทางการติดต่อสื่อสารกับทีม CCD เพื่อขอความช่วยเหลือ-0.pdf]' }
    ]

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, chatgpt_model: str, embedding_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = chatgpt_model
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    def run(self, history: Sequence[dict[str, str]], overrides: dict[str, Any]) -> Any:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        user_q = 'Generate search query for: ' + history[-1]["user"]

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        messages = self.get_messages_from_history(
            self.query_prompt_template,
            self.chatgpt_model,
            history,
            user_q,
            self.query_prompt_few_shots,
            self.chatgpt_token_limit - len(user_q)
            )

        chat_completion = openai.ChatCompletion.create(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
            n=1)

        query_text = chat_completion.choices[0].message.content
        if query_text.strip() == "0":
            query_text = history[-1]["user"] # Use the last user input if we failed to generate a better query

        # Test text from user
        text = history[-1]["user"]
        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            query_vector = openai.Embedding.create(engine=self.embedding_deployment, input=query_text)["data"][0]["embedding"]
        else:
            query_vector = None

         # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            query_text = None

        # Use semantic L2 reranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if overrides.get("semantic_ranker") and has_text:
            r = self.search_client.search(query_text,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,
                                          query_language="en-us",
                                          query_speller="lexicon",
                                          semantic_configuration_name="default",
                                          top=top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
        else:
            r = self.search_client.search(query_text,
                                          filter=filter,
                                          top=top,
                                          vector=query_vector,
                                          top_k=50 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
            
        # else:
        #     r = self.search_client.search(query_text,
        #                                   filter=filter,
        #                                   top=top,
        #                                   vector=query_vector,
        #                                   top_k=50 if query_vector else None,
        #                                   vector_fields="embedding" if query_vector else None)
        
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_override")
        if prompt_override is None:
            system_message = self.system_message_chat_conversation.format(injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            system_message = self.system_message_chat_conversation.format(injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)

        messages = self.get_messages_from_history(
            system_message + "\n\nSources:\n" + content,
            self.chatgpt_model,
            history,
            history[-1]["user"],
            max_tokens=self.chatgpt_token_limit)

        chat_completion = openai.ChatCompletion.create(
            deployment_id=self.chatgpt_deployment,
            model=self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature") or 0.7,
            max_tokens=1500,
            n=1)
        chat_content = chat_completion.choices[0].message.content
        total_tokens = chat_completion.usage.total_tokens

        msg_to_display = '\n\n'.join([str(message) for message in messages])

        return {"data_points": results, "answer": chat_content,"total_token": total_tokens, "thoughts": f"Searched for:<br>{query_text}<br><br>Conversations:<br>" + msg_to_display.replace('\n', '<br>')}

    def get_messages_from_history(self, system_prompt: str, model_id: str, history: Sequence[dict[str, str]], user_conv: str, few_shots = [], max_tokens: int = 1024) -> []:
        message_builder = MessageBuilder(system_prompt, model_id)

        # Add examples to show the chat what responses we want. It will try to mimic any responses and make sure they match the rules laid out in the system message.
        for shot in few_shots:
            message_builder.append_message(shot.get('role'), shot.get('content'))

        user_content = user_conv
        append_index = len(few_shots) + 1

        message_builder.append_message(self.USER, user_content, index=append_index)

        for h in reversed(history[:-1]):
            if h.get("bot"):
                message_builder.append_message(self.ASSISTANT, h.get('bot'), index=append_index)
            message_builder.append_message(self.USER, h.get('user'), index=append_index)
            if message_builder.token_length > max_tokens:
                break

        messages = message_builder.messages
        return messages
