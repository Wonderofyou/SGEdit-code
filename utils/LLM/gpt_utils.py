import base64
import requests
from io import BytesIO
import os
from dotenv import load_dotenv
from openai import OpenAI

# OpenAI API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

class Chat_w_Vision:
    def __init__(self, model_name='gemini-2.5-pro') :
        # Khởi tạo chat session
        self.model = genai.GenerativeModel(model_name)
        self.chat = self.model.start_chat(history=[])
        self.messages = []       # Lưu lịch sử hội thoại (user + model)
        self.gpt_history = []    # Lưu câu trả lời của model

    def _create_message(self, question, image=None):
        """
        Tạo message cho user.
        - Nếu image != None, content là [text, image].
        - Nếu image == None, chỉ là text.
        """
        content = [question, image] if image is not None else question
        return {
            "role": "user",
            "content": content
        }

    def _add_model_message(self, message):
        """Thêm câu trả lời của model vào lịch sử."""
        self.messages.append({"role": "model", "content": message})

    def ask_GPT(self, question, image=None, show_chats=False):
        """
        Gửi câu hỏi tới model:
        - question: text câu hỏi.
        - image: ảnh (có thể None).
        """
        try:
            user_message = self._create_message(question, image)
            self.messages.append(user_message)

            # Gửi tới model
            response = self.chat.send_message(user_message["content"])
            answer = response.text

            self.gpt_history.append(answer)
            self._add_model_message(answer)

            if show_chats:
                print(f"User: {question}")
                if image is not None:
                    print("[Image attached]")
                print("=============================")
                print(f"LLM: {answer}")
                print("=============================")

            return answer
        except Exception as e:
            print(f"Error: {e}")
            return f"Error occurred: {str(e)}"
