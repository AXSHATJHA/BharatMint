from fastapi import FastAPI, HTTPException
from groq import Groq
import os
from dotenv import load_dotenv
import re
from pydantic import BaseModel

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Initialize FastAPI app
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question

        # Step 1: Capture the response from deepseek-r1-distill-llama-70b
        deepseek_response = ""
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            temperature=0.6,
            max_completion_tokens=400,
            top_p=0.95,
            stream=True,
            stop=None,
        )

        for chunk in completion:
            deepseek_response += chunk.choices[0].delta.content or ""

        # Step 2: Process the response to remove unnecessary parts
        cleaned_response = re.sub(r'\[.*?\]', '', deepseek_response).strip()

        # Step 3: Use the cleaned response in the qwen-2.5-32b model
        chat_completion = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=[
                {
                    "role": "user",
                    "content": cleaned_response,
                }
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True,
            stop=None,
        )

        final_response = ""
        for chunks in chat_completion:
            final_response += chunks.choices[0].delta.content or ""

        return {"response": final_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)