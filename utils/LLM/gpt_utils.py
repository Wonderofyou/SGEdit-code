import base64
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from io import BytesIO

# Gemini API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  # Changed from OPENAI_API_KEY
genai.configure(api_key=api_key)

# Function to encode the image
def encode_image(img, size=(512, 512)):
    # Resize the image
    img = img.resize(size)
    # Save the resized image to a byte buffer
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer

class Chat_w_Vision:
    def __init__(self, img) -> None:
        self.image = encode_image(img)
        self.model = genai.GenerativeModel('gemini-2.5-pro')  # Using Gemini Pro with vision
        self.chat_session = None
        self.gpt_history = []  # Keeping the same variable name for compatibility
    
    def ask_GPT(self, question, show_chats=False):
        try:
            if self.chat_session is None:
                # First message with image
                self.image.seek(0)  # Reset buffer position
                pil_image = Image.open(self.image)
                response = self.model.generate_content([question, pil_image])
                
                # Start chat session for follow-up questions
                self.chat_session = self.model.start_chat()
            else:
                # Follow-up messages without image
                response = self.chat_session.send_message(question)
            
            response_text = response.text
            self.gpt_history.append(response_text)
            
            if show_chats:
                print(f"Agent: {question}")
                print("=============================")
                print(f"LLM: {response_text}")
                print("=============================")
                
            return response_text
            
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return None

class Chat:
    def __init__(self) -> None:
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.chat_session = self.model.start_chat()
    
    def ask_GPT(self, question, show_chats=False):
        try:
            response = self.chat_session.send_message(question)
            response_text = response.text
            
            if show_chats:
                print(f"Agent: {question}")
                print("=============================")
                print(f"LLM: {response_text}")
                print("=============================")
                
            return response_text
            
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return None
