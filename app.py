import streamlit as st

from agent import run_agent


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="GuardRail",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("GuardRail")
st.caption("PII Protection Middleware for Agentic AI")

st.info(
    "Customer Support Agent protected by a "
    "LangChain GuardRail Middleware."
)

st.divider()


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("Customer Support Query")

user_input = st.text_area(
    "Enter your message",
    placeholder=(
        "Example:\n"
        "Hi, my name is Harshal. "
        "My email is harshal@gmail.com and "
        "I need help tracking my order."
    ),
    height=150
)


# --------------------------------------------------
# PROCESS REQUEST
# --------------------------------------------------

if st.button(
    "Send to Customer Support Agent",
    type="primary"
):

    if not user_input.strip():

        st.warning(
            "Please enter a customer support message."
        )

        st.stop()

    with st.spinner("Processing request..."):

        try:

            result = run_agent(user_input)

        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )

            st.stop()


    # --------------------------------------------------
    # ORIGINAL PROMPT
    # --------------------------------------------------

    st.divider()

    st.subheader("1. Original User Prompt")

    st.code(
        result.get(
            "original_prompt",
            user_input
        ),
        language="text"
    )


    # --------------------------------------------------
    # PII DETECTION
    # --------------------------------------------------

    st.subheader("2. GuardRail Detection")

    detected_pii = result.get(
        "detected_pii",
        []
    )

    if detected_pii:

        st.write(
            f"**{len(detected_pii)} PII "
            f"entities detected**"
        )

        for entity in detected_pii:

            col1, col2, col3 = st.columns(3)

            with col1:

                st.write(
                    f"**Type:** "
                    f"{entity.get('type', 'UNKNOWN')}"
                )

            with col2:

                st.write(
                    f"**Original:** "
                    f"`{entity.get('value', '')}`"
                )

            with col3:

                st.write(
                    f"**Token:** "
                    f"`{entity.get('token', '')}`"
                )

    else:

        st.success(
            "No PII detected."
        )


    # --------------------------------------------------
    # SANITIZED PROMPT
    # --------------------------------------------------

    st.subheader("3. Sanitized Prompt")

    st.code(
        result.get(
            "sanitized_prompt",
            user_input
        ),
        language="text"
    )

    st.caption(
        "This sanitized prompt is what the "
        "LLM receives."
    )


    # --------------------------------------------------
    # RAW LLM RESPONSE
    # --------------------------------------------------

    st.subheader("4. Raw LLM Response")

    st.code(
        result.get(
            "raw_response",
            ""
        ),
        language="text"
    )

    st.caption(
        "Response received from the agent "
        "before PII restoration."
    )


    # --------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------

    st.subheader("5. Final Customer Response")

    st.success(
        result.get(
            "final_response",
            ""
        )
    )


    # --------------------------------------------------
    # EXECUTION METRICS
    # --------------------------------------------------

    st.divider()

    st.subheader("Execution Metrics")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "PII Masking",
            f"{result.get('masking_latency_ms', 0):.2f} ms"
        )


    with col2:

        st.metric(
            "Agent / LLM",
            f"{result.get('agent_processing_ms', 0):.2f} ms"
        )


    with col3:

        st.metric(
            "PII Unmasking",
            f"{result.get('unmasking_latency_ms', 0):.2f} ms"
        )


    with col4:

        st.metric(
            "Total",
            f"{result.get('latency_ms', 0):.2f} ms"
        )