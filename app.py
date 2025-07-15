from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel

load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Additional conversation context, if any"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Hrishikesh Kulkarni"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load resume
        resume_path = os.path.join(current_dir, "resume.pdf")
        reader = PdfReader(resume_path)
        self.resume = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.resume += text
        
        # Load summary
        summary_path = os.path.join(current_dir, "summary.txt")
        with open(summary_path, "r") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results

    def system_prompt(self):
        prompt = f"""You are acting as {self.name}. You are answering questions on {self.name}'s website,
particularly questions related to {self.name}'s career, background, skills and experience.
Be professional and engaging. If you don't know the answer, use the record_unknown_question tool.
Encourage users to share their email and use record_user_details when they do.

## Summary:
{self.summary}

## Resume:
{self.resume}

With this context, please chat with the user, staying in character as {self.name}.
"""
        return prompt

    def convert_history(self, history):
        messages = []
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        return messages

    def format_history(self, history):
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

    def evaluator_system_prompt(self):
        return f"""You are an evaluator that decides whether a response is acceptable.
The Agent is playing the role of {self.name} and is representing {self.name} professionally.
Here's the context:

## Summary:
{self.summary}

## Resume:
{self.resume}

Evaluate the latest response. Reply with valid JSON in this format:
{{"is_acceptable": true, "feedback": "..."}}
"""

    def evaluate(self, reply, message, history):
        history_string = self.format_history(history)
        user_prompt = f"""Question: {message}

Conversation History:
{history_string}

Agent's Response:
{reply}
"""
        messages = [
            {"role": "system", "content": self.evaluator_system_prompt()},
            {"role": "user", "content": user_prompt}
        ]
        response = self.openai.chat.completions.create(
            model="gpt-4.1-nano", messages=messages
        )
        response_content = response.choices[0].message.content
        try:
            data = json.loads(response_content)
            return Evaluation(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse evaluation response: {response_content}") from e

    def rerun(self, reply, message, history, feedback):
        updated_system_prompt = self.system_prompt() + f"""
## Previous answer rejected
You just tried to reply, but quality control rejected your answer.

## Your attempted answer:
{reply}

## Reason for rejection:
{feedback}
"""
        messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]
        response = self.openai.chat.completions.create(
            model="gpt-4.1-nano", messages=messages
        )
        return response.choices[0].message.content

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        
        # First response loop (tools or not)
        while True:
            responses = self.openai.chat.completions.create(
                model="gpt-4o-mini", messages=messages, tools=tools
            )
            finish_reason = responses.choices[0].finish_reason

            if finish_reason == 'tool_calls':
                response = responses.choices[0].message
                tool_calls = response.tool_calls
                messages.append(response)
                results = self.handle_tool_call(tool_calls)
                messages.extend(results)
            else:
                result = responses.choices[0].message.content
                break

        # Evaluation loop
        while True:
            evaluation = self.evaluate(result, message, history)
            if evaluation.is_acceptable:
                print("Evaluation: Acceptable")
                return result
            else:
                print("Evaluation: Rejected")
                result = self.rerun(result, message, history, evaluation.feedback)

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
