
import os, json
from typing import TypedDict, Annotated, Literal, Optional

import joblib
import numpy as np
import pandas as pd

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
MODEL_PATH = "./artifacts/credit_model.pkl"

MAX_ITERATIONS = 5
DB_DIR = "./artifacts/chroma_db"         

embedder = OllamaEmbeddings(model="nomic-embed-text", num_gpu=0)
store = Chroma(collection_name="lending_docs", embedding_function=embedder, persist_directory=DB_DIR)
try:
    credit_model = joblib.load(MODEL_PATH)     
    MODEL_ERROR = None
except Exception as e:
    credit_model, MODEL_ERROR = None, str(e)

LOAN_INTENTS = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]

@tool
def search_lending_documents(query: str) -> str:
    """Search the lending documents and return relevant passages as JSON.
    Topics: personal loan requirements, credit scores needed, types of loans (secured, unsecured,
    payday, title, pawnshop), debt-to-income ratio, creditworthiness (the five C's of credit),
    improving your credit score, debt consolidation, and credit discrimination / fair lending rights.

    Args:
        query: a short search question in English.
    """
    if not query.strip():
        return json.dumps({"error": "Empty query."})
    try:
        docs = store.as_retriever(search_kwargs={"k": 3}).invoke(query)
        passages = [{"text": d.page_content,
                     "source": os.path.basename(d.metadata["source"]),
                     "page": d.metadata["page"] + 1} for d in docs]
        return json.dumps({"query": query, "passages": passages})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def predict_credit_risk(person_age: Optional[int] = None, person_income: Optional[float] = None,
                        loan_amnt: Optional[float] = None, loan_intent: Optional[str] = None,
                        person_home_ownership: Optional[str] = None, person_emp_length: Optional[float] = None,
                        loan_grade: Optional[str] = None, loan_int_rate: Optional[float] = None,
                        cb_person_default_on_file: Optional[str] = None,
                        cb_person_cred_hist_length: Optional[int] = None) -> str:
    """Predict the user's OWN credit risk (probability of not repaying the loan) with a machine-learning model.
    Use it when the user gives their financial details or asks about their own risk.
    Required: person_age, person_income, loan_amnt, loan_intent. The other fields are optional.

    Args:
        person_age: age in years.
        person_income: yearly income in dollars.
        loan_amnt: requested loan amount in dollars.
        loan_intent: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT or DEBTCONSOLIDATION.
        person_home_ownership: RENT, OWN, MORTGAGE or OTHER.
        person_emp_length: years of employment.
        loan_grade: loan grade from A (best) to G (worst).
        loan_int_rate: interest rate in percent, e.g. 11.5.
        cb_person_default_on_file: "Y" if the person defaulted on a loan before, otherwise "N".
        cb_person_cred_hist_length: years of credit history.
    """
    if credit_model is None:
        return json.dumps({"error": f"ML model not available: {MODEL_ERROR}"})

    required = {"person_age": person_age, "person_income": person_income,
                "loan_amnt": loan_amnt, "loan_intent": loan_intent}
    missing = [k for k, v in required.items() if v in (None, "")]
    if missing:                                              # agent should ask the user for these
        return json.dumps({"error": "Missing required information", "missing_fields": missing})

    loan_intent = str(loan_intent).upper().replace(" ", "").replace("_", "")
    if loan_intent not in LOAN_INTENTS:
        return json.dumps({"error": f"Invalid loan_intent '{loan_intent}'", "allowed": LOAN_INTENTS})
    if not (18 <= person_age <= 100) or person_income <= 0 or loan_amnt <= 0:
        return json.dumps({"error": "Invalid values: age must be 18-100, income and loan amount must be > 0"})

    try:
        row = {
            "person_age": person_age, "person_income": person_income,
            "person_home_ownership": person_home_ownership.upper() if person_home_ownership else np.nan,
            "person_emp_length": person_emp_length if person_emp_length is not None else np.nan,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade.upper() if loan_grade else np.nan,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate if loan_int_rate is not None else np.nan,
            "loan_percent_income": round(loan_amnt / person_income, 2),
            "cb_person_default_on_file": cb_person_default_on_file.upper() if cb_person_default_on_file else np.nan,
            "cb_person_cred_hist_length": cb_person_cred_hist_length if cb_person_cred_hist_length is not None else np.nan,
        }
        prob = float(credit_model.predict_proba(pd.DataFrame([row]))[0][1])     # probability of default
        level = "Low" if prob < 0.2 else "Medium" if prob < 0.5 else "High"
        filled = [k for k, v in row.items() if isinstance(v, float) and np.isnan(v)]
        return json.dumps({"risk_level": level, "probability_of_default": round(prob, 2),
                           "loan_percent_income": row["loan_percent_income"],
                           "fields_filled_with_typical_values": filled,
                           "note": "Educational estimate only, not a lending decision."})
    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {e}"})

tools = [search_lending_documents,predict_credit_risk]         
TOOLS = {t.name: t for t in tools}



llm = ChatOllama(model="qwen3.5:4b", temperature=0, reasoning=False, num_ctx=4096, num_gpu=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are an educational Credit Risk Assessment Assistant. You never approve or reject loans.
Tools:
- predict_credit_risk: estimate the USER'S OWN credit risk (Low/Medium/High) from their financial details.
- search_lending_documents: general lending guides (credit scores, loan types, DTI, improving credit, rights).
- query_loan_graph: the bank's loan products, their conditions, required documents and risk policies.
Rules:
- Choose the tools the question needs. You may use several tools, one after another.
- Use ONLY the tool results. Never invent numbers, facts or sources.
- If the tools don't have the answer, say so and stop. Do not add facts from your own knowledge.
- If predict_credit_risk says information is missing, ask the user for exactly those fields.
- After a risk prediction, you may call query_loan_graph with query_type "risk" to explain what that risk level means,
  and with query_type "intent" to find the matching loan product.
- If the request is unclear, ask the user ONE short question.
- If a tool returns an error, fix the input and try again, or explain the problem.
- Stop calling tools when you have enough information and give the final answer."""

FINAL_PROMPT = ("This is the last iteration. Do not call tools. Give the final answer using ONLY the tool "
                "results above, and clearly say what you found and what is still unresolved.")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  
    iteration: int                           



def llm_node(state: AgentState):
    iteration = state["iteration"] + 1
    messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
    try:
        if iteration < MAX_ITERATIONS:
            response = llm_with_tools.invoke(messages)
        else:                                         
            response = llm.invoke(messages + [HumanMessage(FINAL_PROMPT)])
    except Exception as e:
        response = AIMessage(content=f"Sorry, the model failed: {e}")
    for call in response.tool_calls:
        print(f"\n[iteration {iteration}] Calling tool: {call['name']}")
        

    if not response.tool_calls:
        print(f"\n[iteration {iteration}] Generating final answer...")


    return {"messages": [response], "iteration": iteration}


def tool_node(state: AgentState):
    results = []
    for call in state["messages"][-1].tool_calls:
        try:
            if call["name"] not in TOOLS:
                raise ValueError(f"Unknown tool: {call['name']}")
            output = TOOLS[call["name"]].invoke(call["args"])
        except Exception as e:                          
            output = json.dumps({"error": str(e)})
        results.append(ToolMessage(content=output, tool_call_id=call["id"]))
    return {"messages": results}


def should_continue(state: AgentState):
    return "tools" if state["messages"][-1].tool_calls else END     


graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")
agent = graph.compile(checkpointer=MemorySaver())       


def main():
    session = 1
    print("-" * 70)
    print(" Credit Risk Assessment Assistant ")
    print(" Type your question.  /new = new conversation   exit = quit")
    print("-" * 70)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):            
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        if question == "/new":
            session += 1
            print("Started a new conversation.")
            continue

    
        try:
            state = agent.invoke({"messages": [HumanMessage(question)], "iteration": 0},
                                 {"configurable": {"thread_id": str(session)}})
            answer = state["messages"][-1].content
        except Exception as e:
            answer = f"Agent error: {e}"
        print(f"\nAssistant: {answer}")
   

    print("Goodbye!")


if __name__ == "__main__":
    main()