# # models.py
# import config as c
# import typing as t
# import database as d
# from langchain_community.llms import Ollama
# from langchain_core.output_parsers import StrOutputParser

# class UniversityChatbot:
#     def __init__(self, user_id: str, database: d.Database):
#         self.user_id = user_id
#         self.db = database
#         self.llm = Ollama(model=c.LLM_MODEL, base_url=c.OLLAMA_BASE_URL, temperature=0.1)
#         self.parser = StrOutputParser()

#     def _get_system_prompt(self) -> str:
#         """Generate a system prompt with user context."""
#         demographics = self.db.get_user_demographics(self.user_id)
#         if not demographics:
#             return "You are a helpful university student assistant."

#         demographics_str = f"""
#         Student Name: {demographics.get('name', 'Unknown')}
#         Age: {demographics.get('age', 'Unknown')}
#         Education: {demographics.get('education', 'Unknown')}
#         Hometown: {demographics.get('hometown', 'Unknown')}
#         Email: {demographics.get('email', 'Unknown')}
#         """
#         system_prompt = f"""You are a helpful university student assistant chatbot.

# Your responsibilities:
# 1. Answer questions about university resources, programs, and facilities
# 2. Help students find relevant information from the university database
# 3. Be personalized and considerate of the student's profile
# 4. Extract and track the student's interests from conversations
# 5. Provide accurate, helpful, and friendly responses

# STUDENT PROFILE:
# {demographics_str}

# GUIDELINES:
# - Be conversational, supportive, and encouraging
# - If you don't have specific information, acknowledge it and suggest resources
# - When providing university information, cite the source clearly
# - Be concise but informative in your responses
# - Show genuine interest in helping the student"""
        
#         return system_prompt

#     def retrieve_university_info(self, query: str, k: int = 3) -> t.List[t.Dict[str, t.Any]]:
#         """Retrieve relevant university information from vector database."""
#         return self.db.search_university_info(query, k=k)

#     def format_context(self, university_info: t.List[t.Dict[str, t.Any]]) -> str:
#         """Format university information for the LLM context."""
#         if not university_info:
#             return "No relevant university information found in the database."
        
#         formatted = "RELEVANT UNIVERSITY INFORMATION:\n"
#         for i, item in enumerate(university_info, 1):
#             title = item.get('metadata', {}).get('title', 'Information')
#             content = item.get('content', '')
#             page = item.get('metadata', {}).get('page', 'unknown')
            
#             formatted += f"\n{i}. {title}\n"
#             formatted += f"   Content: {content}\n"
#             formatted += f"   Source: {page}\n"
#         return formatted

#     def extract_interests(self, user_message: str) -> t.List[str]:
#         """Extract potential interests from the user's message only."""
#         extraction_prompt = (
#             "Analyze the following student message and identify any new interests, "
#             "preferences, or topics the student is interested in.\n\n"
#             f"Student message: {user_message}\n\n"
#             "Extract interests as a simple list (one per line, starting with a dash).\n"
#             "If no clear interests can be extracted, respond with: NONE\n\n"
#             "Extracted interests:"
#         )
#         try:
#             response = self.llm.invoke(extraction_prompt)
#             interests = []
#             if response.strip() != "NONE" and "NONE" not in response:
#                 lines = response.strip().split('\n')
#                 for line in lines:
#                     line = line.strip()
#                     if line and line.startswith('-'):
#                         interest = line[1:].strip()
#                         if interest:
#                             interests.append(interest)
#                     elif line and not line.startswith('#'):
#                         interests.append(line)
#             return [i for i in interests if len(i.strip()) > 0][:5]  # Limit to 5 interests
#         except Exception as e:
#             print(f"Error extracting interests: {e}")
#             return []

#     def generate_response(self, user_message: str) -> str:
#         """Generate a chatbot response to the user message."""
#         try:
#             # Retrieve relevant university information
#             university_info = self.retrieve_university_info(user_message)
#             context = self.format_context(university_info)
#             # Build the full prompt
#             system_prompt = self._get_system_prompt()
#             full_prompt = f"""{system_prompt}

# {context}

# Student Question: {user_message}

# Assistant Response:"""
#             # Generate response using Ollama
#             response = self.llm.invoke(full_prompt)
#             return response.strip()
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return "I apologize, but I encountered an error processing your question. Please try again."

# models.py
import config as c
import typing as t
import database as d
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

class UniversityChatbot:
    def __init__(self, user_id: str, database: d.Database):
        self.user_id = user_id
        self.db = database
        self.llm = Ollama(model=c.LLM_MODEL, base_url=c.OLLAMA_BASE_URL, temperature=0.7)
        self.parser = StrOutputParser()
        self.conversation_context = []  # Store recent conversation turns

    def _get_system_prompt(self) -> str:
        """Generate a system prompt with user context."""
        demographics = self.db.get_user_demographics(self.user_id)
        if not demographics:
            return "You are a helpful university student assistant."

        name = demographics.get('name', 'Unknown').split()[0]  # Use first name only
        
        system_prompt = f"""You are a friendly and helpful university assistant chatbot talking to {name}, a {demographics.get('education', 'student')}.

IMPORTANT CONVERSATION RULES:
- Maintain natural conversation flow - don't repeat greetings if you're mid-conversation
- Remember context from previous messages in this chat session
- When user gives short responses like "yes", "no", "tell me more", refer back to what you were discussing
- Be warm but not overly enthusiastic - match the user's energy level
- Don't repeat information unnecessarily
- Keep responses concise unless the user asks for detailed information
- NEVER start every response with "Hi {name}!" - only greet at the start of conversation

YOUR ROLE:
- Answer questions about university resources, programs, and facilities
- Help students find information from the university database
- Provide personalized assistance based on their profile
- Be conversational and supportive

RESPONSE STYLE:
- Natural and conversational, not formal or robotic
- Brief responses for simple questions (1-2 sentences)
- Detailed responses only when specifically requested
- Use context from previous messages to maintain conversation continuity"""
        
        return system_prompt

    def _format_conversation_history(self, limit: int = 4) -> str:
        """Format recent conversation history for context."""
        if not self.conversation_context:
            return ""
        
        # Get the last few turns
        recent = self.conversation_context[-limit:]
        formatted = "\nRECENT CONVERSATION:\n"
        for turn in recent:
            formatted += f"Student: {turn['user']}\n"
            formatted += f"Assistant: {turn['assistant']}\n"
        return formatted

    def retrieve_university_info(self, query: str, k: int = 3) -> t.List[t.Dict[str, t.Any]]:
        """Retrieve relevant university information from vector database."""
        return self.db.search_university_info(query, k=k)

    def format_context(self, university_info: t.List[t.Dict[str, t.Any]]) -> str:
        """Format university information for the LLM context."""
        if not university_info:
            return ""
        
        formatted = "\nRELEVANT UNIVERSITY INFORMATION:\n"
        for i, item in enumerate(university_info, 1):
            title = item.get('metadata', {}).get('title', 'Information')
            content = item.get('content', '')
            page = item.get('metadata', {}).get('page', 'unknown')
            
            formatted += f"\n[Source {i}: {title}]\n{content}\n"
        return formatted

    def extract_interests(self, user_message: str) -> t.List[str]:
        """Extract potential interests from the user's message only."""
        # Skip extraction for very short messages or common responses
        if len(user_message.split()) < 3 or user_message.lower() in ['yes', 'no', 'ok', 'sure', 'thanks', 'thank you']:
            return []
        
        extraction_prompt = (
            "Analyze this student message and extract ONLY clear, specific interests or topics.\n\n"
            f"Message: {user_message}\n\n"
            "Rules:\n"
            "- Extract only genuine interests (hobbies, academic topics, activities)\n"
            "- Ignore questions or general statements\n"
            "- Be specific (not 'housing' but 'on-campus housing' if they want to live there)\n"
            "- Format: one interest per line starting with '-'\n"
            "- If no clear interests, respond with: NONE\n\n"
            "Interests:"
        )
        try:
            response = self.llm.invoke(extraction_prompt)
            interests = []
            if response.strip() != "NONE" and "NONE" not in response.upper():
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and line.startswith('-'):
                        interest = line[1:].strip()
                        if interest and len(interest) > 3:
                            interests.append(interest)
            return interests[:3]  # Limit to 3 interests per message
        except Exception as e:
            print(f"Error extracting interests: {e}")
            return []

    def generate_response(self, user_message: str) -> str:
        """Generate a chatbot response to the user message."""
        try:
            # Retrieve relevant university information only for substantive questions
            university_info = []
            if len(user_message.split()) > 2 and user_message.lower() not in ['yes', 'no', 'sure', 'ok']:
                university_info = self.retrieve_university_info(user_message)
            
            context = self.format_context(university_info)
            conversation_history = self._format_conversation_history()
            
            # Build the full prompt
            system_prompt = self._get_system_prompt()
            
            full_prompt = f"""{system_prompt}
{conversation_history}
{context}

Current Student Message: {user_message}

Assistant (respond naturally, maintaining conversation flow):"""
            # Generate response using Ollama
            response = self.llm.invoke(full_prompt)
            response = response.strip()
            
            # Store in conversation context (keep last 6 turns)
            self.conversation_context.append({
                'user': user_message,
                'assistant': response
            })
            if len(self.conversation_context) > 6:
                self.conversation_context.pop(0)
            
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error. Could you rephrase your question?"
