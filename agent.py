import os
import time

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.tools import tool

from middleware import GuardRailMiddleware


load_dotenv()


# --------------------------------------------------
# TRUSTED CUSTOMER SUPPORT TOOL
# --------------------------------------------------

@tool
def get_order_status(email: str) -> str:
    """
    Look up a customer's order status using their email.
    This represents a trusted backend/customer database.
    """

    # Simulated customer database
    orders = {
        "harshal@gmail.com": {
            "order_id": "AP001254",
            "status": "Shipped",
            "estimated_delivery": "August 20, 2026"
        },
        "rahul@gmail.com": {
            "order_id": "AP001255",
            "status": "Processing",
            "estimated_delivery": "August 22, 2026"
        }
    }

    customer = orders.get(email)

    if not customer:
        return "No order was found for this email address."

    return (
        f"Order {customer['order_id']} is currently "
        f"{customer['status']}. "
        f"Estimated delivery: "
        f"{customer['estimated_delivery']}."
    )


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# --------------------------------------------------
# AGENT
# --------------------------------------------------

agent = create_agent(
    model=llm,

    tools=[
        get_order_status
    ],

    system_prompt=(
        "You are a customer support agent.\n\n"

        "You can check customer orders using the "
        "get_order_status tool.\n\n"

        "Sensitive information such as names, emails, "
        "and phone numbers may appear as placeholders "
        "such as [PERSON_0], [EMAIL_0], or [PHONE_0].\n\n"

        "Never ask the customer to reveal sensitive "
        "information again if it has already been provided.\n\n"

        "When an order status is requested and an email "
        "placeholder is available, use that placeholder "
        "with the get_order_status tool."
    ),

    middleware=[
        GuardRailMiddleware()
    ]
)


# --------------------------------------------------
# RUN AGENT
# --------------------------------------------------

def run_agent(user_input: str):

    total_start = time.perf_counter()

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    })

    total_latency_ms = (
        time.perf_counter() - total_start
    ) * 1000

    masking_latency_ms = result.get(
        "masking_latency_ms",
        0.0
    )

    unmasking_latency_ms = result.get(
        "unmasking_latency_ms",
        0.0
    )

    agent_processing_ms = max(
        0.0,
        total_latency_ms
        - masking_latency_ms
        - unmasking_latency_ms
    )

    final_response = result.get(
        "final_response"
    )

    if not final_response:

        final_response = (
            result["messages"][-1].content
        )

    return {
        "original_prompt": user_input,

        "detected_pii": result.get(
            "detected_pii",
            []
        ),

        "sanitized_prompt": result.get(
            "sanitized_prompt",
            user_input
        ),

        "raw_response": result.get(
            "raw_response",
            final_response
        ),

        "final_response": final_response,

        "masking_latency_ms": masking_latency_ms,

        "agent_processing_ms": agent_processing_ms,

        "unmasking_latency_ms": unmasking_latency_ms,

        "latency_ms": total_latency_ms
    }


# --------------------------------------------------
# TERMINAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    user_input = input("You: ")

    result = run_agent(user_input)

    print("\n========== ORIGINAL ==========")
    print(result["original_prompt"])

    print("\n========== DETECTED PII ==========")
    print(result["detected_pii"])

    print("\n========== SANITIZED ==========")
    print(result["sanitized_prompt"])

    print("\n========== RAW RESPONSE ==========")
    print(result["raw_response"])

    print("\n========== FINAL RESPONSE ==========")
    print(result["final_response"])

    print("\n========== PERFORMANCE ==========")

    print(
        f"PII masking: "
        f"{result['masking_latency_ms']:.2f} ms"
    )

    print(
        f"Agent / LLM: "
        f"{result['agent_processing_ms']:.2f} ms"
    )

    print(
        f"PII unmasking: "
        f"{result['unmasking_latency_ms']:.2f} ms"
    )

    print(
        f"Total: "
        f"{result['latency_ms']:.2f} ms"
    )