import base64
import requests
from io import BytesIO
import os
from dotenv import load_dotenv
from openai import OpenAI

# OpenAI API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Function to encode the image
def encode_image(img, size=(512, 512)):
    # Resize the image
    img = img.resize(size)

    # Save the resized image to a byte buffer
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    # Encode the image
    return base64.b64encode(buffer.read()).decode('utf-8')


class Chat_w_Vision:
    def __init__(self, img) -> None:
        self.base64_image = encode_image(img)
        self.headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {api_key}"
        }
        self.messages = []
        self.gpt_history = []
    
    def create_initial_message(self, question):
        new_message = {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": question
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{self.base64_image}"
                }
              }
            ]
          }
        self.messages.append(new_message)
    
    def create_follow_message(self, question):
        new_message = {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": question
                }
            ]
        }
        self.messages.append(new_message)
    
    def add_message(self, message):
        self.messages.append(message)
    
    def ask_GPT(self, question, show_chats=False):
        if len(self.messages) == 0:
            self.create_initial_message(question)
        else:
            self.create_follow_message(question)
        
        self.payload = {
          "model": "gpt-4-vision-preview",
          "messages": self.messages,
          "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=self.payload).json()
        gpt_message = response["choices"][0]["message"]

        if "choices" not in response:
            print("GPT refuse to reply. You might not have sufficient money in your account.")
            return None
        
        self.gpt_history.append(gpt_message["content"])
        self.add_message(gpt_message)

        if show_chats:
            print(f"Agent: {question}")
            print("=============================")
            print(f"LLM: {gpt_message['content']}")
            print("=============================")
            
        return gpt_message["content"]
    

class Chat:
    def __init__(self) -> None:
      self.client = OpenAI()
      self.messages = []

    def add_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def ask_GPT(self, question, show_chats=False):
        self.add_message(question)
        response = self.client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=self.messages,
            )
        gpt_message = response.choices[0].message
        self.messages.append(gpt_message)

        if show_chats:
            print(f"Agent: {question}")
            print("=============================")
            print(f"LLM: {gpt_message.content}")
            print("=============================")

        return gpt_message.content