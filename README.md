# PII Guardrail Middleware for Agentic AI

## Demo

Here is a demonstration of the PII Guardrail Middleware in action.

https://github.com/user-attachments/assets/5e226b2e-cbbb-4c65-9d09-1af63d6d8f0c

A PII protection middleware built with LangChain for protecting sensitive information in agentic AI applications.

This project demonstrates how sensitive user information can be detected and masked before reaching an LLM, while allowing the original information to be restored when required.

## Overview

The project uses a simple **Customer Support Agent** that allows users to check their order status.

The guardrail sits between the user and the agent:

**User Prompt → PII Detection → PII Masking → LLM / Agent → Tool Call → PII Restoration → Final Response**

For example:

**Original Prompt**

> My name is Harshal and my email is harshal@gmail.com. Please check my order.

**Sanitized Prompt**

> My name is `[PERSON_0]` and my email is `[EMAIL_0]`. Please check my order.

The sanitized prompt is passed to the LLM instead of the original PII.

## Features

- PII detection using GLiNER
- Regex-based email and phone detection
- Automatic PII masking
- Placeholder-based anonymization
- PII restoration after agent processing
- Trusted tool boundary for restoring PII
- Customer order-status tool
- Latency measurement
- Streamlit interface
- LangChain `AgentMiddleware` integration

## PII Detection

| PII Type | Detection Method |
|---|---|
| Person | GLiNER |
| Location | GLiNER |
| Email | Regex |
| Phone | Regex |

Detected information is replaced with placeholders such as:

```text
[PERSON_0]
[LOCATION_0]
[EMAIL_0]
[PHONE_0]
```

The original values are stored in an internal mapping and restored only when required.

## Architecture

```text
                    User
                     |
                     v
              +-------------+
              | PII Guardrail|
              +-------------+
                     |
              PII Detection
             GLiNER + Regex
                     |
                     v
                PII Masking
                     |
                     v
              Sanitized Prompt
                     |
                     v
              +-------------+
              | LLM / Agent |
              +-------------+
                     |
                     v
                  Tool Call
                     |
                     v
           Trusted Tool Boundary
                     |
              PII Restoration
                     |
                     v
             Final Response
                     |
                     v
                    User
```

## Performance

In the current local setup, the observed latency is approximately:

- **PII masking:** ~180 ms
- **LLM / Agent processing:** ~1.5 seconds
- **PII unmasking:** ~0.02 ms

Actual latency may vary depending on hardware, input size, model, and network conditions.

## Tech Stack

- Python
- LangChain
- LangGraph
- LangChain Groq
- GLiNER
- Groq
- Streamlit
- Regular Expressions

## Project Structure

```text
guardrails_demo/
│
├── app.py
├── agent.py
├── middleware.py
├── anonymizer.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Harshal-Ug/guardrails-demo.git
cd guardrails-demo
```

### 2. Create the environment

```bash
conda create -n guardrails_demo python=3.12
conda activate guardrails_demo
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

## Future Work

- Prompt injection detection
- Tool-call validation
- Secret detection
- Output validation
- Agent iteration limits
- Token and cost controls
- Additional PII entity types
- PII detection performance optimization

## Repository

[GitHub Repository](https://github.com/Harshal-Ug/guardrails-demo)

## Author

**Harshal Upaganlawar**