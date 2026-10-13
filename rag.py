from dotenv import load_dotenv

load_dotenv()  # must run before the LangChain imports so tracing picks up the keys

import openai
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable

from guardrails import check_nemo_input, validate_answer, validate_question

HISTORY_SIZE = 6

SYSTEM = """You are an assistant for Tamil Nadu agriculture schemes.
Answer ONLY using the context below. If the answer is not in the context, say you
could not find it in the documents. Be clear and concise, and mention eligibility,
subsidy amounts, and how to apply when relevant. Reply in the same language as the
latest question, or in the language the user asks for (for example "tell me in English").

Context:
{context}"""

# ---- loaded once at startup ----
_emb = OpenAIEmbeddings(model="text-embedding-3-small")
_vs = FAISS.load_local("faiss_index", _emb, allow_dangerous_deserialization=True)
retriever = _vs.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True, timeout=60, max_retries=2)
rewriter = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30, max_retries=2)


def friendly_error(e):
    if isinstance(e, openai.AuthenticationError):
        return "The API key is invalid. Please check OPENAI_API_KEY."
    if isinstance(e, openai.RateLimitError):
        return "The service is busy or the API credits have run out. Please try again shortly."
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return "Could not reach the AI service. Please check the internet connection and retry."
    return "Sorry, something went wrong. Please try again in a moment."


def rewrite_query(question: str, history: list) -> str:
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-HISTORY_SIZE:])
    prompt = (
        "Given the conversation and the latest message, write ONE standalone search query "
        "in English that includes the scheme name and topic being discussed. "
        "Output only the query.\n\n"
        f"Conversation:\n{convo}\n\nLatest message: {question}"
    )
    try:
        return rewriter.invoke(prompt).content.strip() or question
    except Exception:
        return question


@traceable(name="tn_agri_rag", run_type="chain")
def run_rag(question: str, history: list):
    """Generator: runs input rails, yields token events, checks output rails, and yields sources."""

    # 1. Structural / Length checks (Fast & Free)
    struct_check = validate_question(question)
    if not struct_check.allowed:
        yield {"type": "token", "data": struct_check.reason}
        yield {"type": "sources", "data": []}
        return

    # 2. NeMo Semantic Input Check (Off-topic, jailbreak prevention)
    nemo_check = check_nemo_input(question)
    if not nemo_check.allowed:
        yield {"type": "token", "data": nemo_check.reason}
        yield {"type": "sources", "data": []}
        return

    # 3. Retrieval & Synthesis
    search_query = rewrite_query(question, history)
    docs = retriever.invoke(search_query)
    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'Unknown')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

    messages = [("system", SYSTEM.format(context=context))]
    messages += [(m["role"], m["content"][:800]) for m in history[-HISTORY_SIZE:]]
    messages.append(("human", question))

    full_response = []
    try:
        for chunk in llm.stream(messages):
            if chunk.content:
                full_response.append(chunk.content)
                yield {"type": "token", "data": chunk.content}
    except Exception as e:
        yield {"type": "token", "data": friendly_error(e)}
        yield {"type": "sources", "data": []}
        return

    # 4. Output Guardrail (Secret keys / credentials check)
    generated_text = "".join(full_response)
    out_check = validate_answer(generated_text)
    if not out_check.allowed:
        yield {
            "type": "token",
            "data": "\n\n[Response flagged and retracted by security guardrail.]",
        }
        yield {"type": "sources", "data": []}
        return

    sources = sorted(
        {
            f"{d.metadata.get('source', 'Unknown')} (p.{d.metadata.get('page', '?')})"
            for d in docs
        }
    )
    yield {"type": "sources", "data": sources}