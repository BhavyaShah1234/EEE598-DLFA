# # main.py
# import sys
# import models as m
# import argparse as g
# import database as d
# import authenticator as a

# def initialize_application() -> d.Database:
#     """Initialize the application and database."""
#     print("Initializing chatbot application...")
#     db = d.Database()
#     db.populate_sample_data()
#     print("Database initialized successfully.")
#     return db

# def display_user_info(session_data: dict) -> None:
#     """Display user information in a formatted way."""
#     print("\n" + "="*60)
#     print("USER SESSION LOADED")
#     print("="*60)
#     demographics = session_data.get('demographics', {})
#     print(f"\nWelcome, {demographics.get('name', 'Student')}!")
#     print(f"Age: {demographics.get('age')}")
#     print(f"Field of Study: {demographics.get('education')}")
#     print(f"Hometown: {demographics.get('hometown')}")
#     print(f"Email: {demographics.get('email')}")
    
#     interests = session_data.get('interests', [])
#     if interests:
#         print(f"\nRecorded Interests:")
#         for interest in interests:
#             print(f"  • {interest}")
    
#     print("\n" + "="*60)
#     print("Type 'exit' or 'quit' to end the chat.")
#     print("="*60 + "\n")

# def run_chatbot(user_id: str, db: d.Database, chatbot: m.UniversityChatbot) -> None:
#     """Run the main chatbot loop."""
#     print("\nChatbot ready. Ask me anything about the university!\n")
#     while True:
#         try:
#             user_input = input("You: ").strip()
#             if user_input.lower() in ['exit', 'quit', 'bye']:
#                 print("\nAssistant: Goodbye! Have a great day!")
#                 break
#             if not user_input:
#                 continue
#             # Add user message to database
#             db.add_chat_message(user_id, "user", user_input)
#             # Generate response
#             print("\nAssistant: ", end="", flush=True)
#             response = chatbot.generate_response(user_input)
#             print(response)
#             print()
#             # Add assistant message to database
#             db.add_chat_message(user_id, "assistant", response)
#         except KeyboardInterrupt:
#             print("\n\nChatbot session interrupted.")
#             break
#         except Exception as e:
#             print(f"\nError: {e}")
#             print("Please try again.\n")

# def main():
#     """Main entry point for the application."""
#     parser = g.ArgumentParser(description="University Student Chatbot")
#     parser.add_argument("--user", type=str, help="User ID (user1, user2, or user3)", required=True)
#     args = parser.parse_args()
#     user_id = args.user
#     # Initialize database
#     db = initialize_application()
#     # Authenticate user
#     authenticator = a.UserAuthenticator(db)
#     if not authenticator.is_valid_user(user_id):
#         print(f"Error: Invalid user ID '{user_id}'")
#         print("Available users:")
#         for uid, name in authenticator.list_available_users().items():
#             print(f"  • {uid}: {name}")
#         sys.exit(1)
#     # Get user session data
#     session_data = authenticator.get_user_session_data(user_id)
#     # Display user information
#     display_user_info(session_data)
#     # Initialize chatbot
#     try:
#         chatbot = m.UniversityChatbot(user_id, db)
#         print("Chatbot model loaded successfully.\n")
#     except Exception as e:
#         print(f"Error: Could not initialize chatbot model: {e}")
#         print("Make sure Ollama is running with 'gemma3' and 'nomic-embed-text' models.")
#         sys.exit(1)
#     # Run chatbot
#     run_chatbot(user_id, db, chatbot)
#     print("\nThank you for using the University Chatbot!")

# if __name__ == "__main__":
#     main()

# main.py
import argparse
import sys
from pathlib import Path
from database import Database
from authenticator import UserAuthenticator
from models import UniversityChatbot


def initialize_application() -> Database:
    """Initialize the application and database."""
    print("Initializing chatbot application...")
    db = Database()
    db.populate_sample_data()
    print("Database initialized successfully.")
    return db


def display_user_info(session_data: dict) -> None:
    """Display user information in a formatted way."""
    print("\n" + "="*60)
    print("USER SESSION LOADED")
    print("="*60)
    
    demographics = session_data.get('demographics', {})
    print(f"\nWelcome, {demographics.get('name', 'Student')}!")
    print(f"Age: {demographics.get('age')}")
    print(f"Field of Study: {demographics.get('education')}")
    print(f"Hometown: {demographics.get('hometown')}")
    print(f"Email: {demographics.get('email')}")
    
    interests = session_data.get('interests', [])
    if interests:
        print(f"\nRecorded Interests:")
        for interest in interests:
            print(f"  • {interest}")
    
    print("\n" + "="*60)
    print("Type 'exit' or 'quit' to end the chat.")
    print("="*60 + "\n")


def run_chatbot(user_id: str, db: Database, chatbot: UniversityChatbot) -> None:
    """Run the main chatbot loop."""
    print("\nChatbot ready. Ask me anything about the university!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nAssistant: Goodbye! Have a great day!")
                break
            
            if not user_input:
                continue
            
            # Add user message to database
            db.add_chat_message(user_id, "user", user_input)
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            response = chatbot.generate_response(user_input)
            print(response)
            print()
            
            # Add assistant message to database
            db.add_chat_message(user_id, "assistant", response)

        except KeyboardInterrupt:
            print("\n\nChatbot session interrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="University Student Chatbot")
    parser.add_argument(
        "--user",
        type=str,
        help="User ID (user1, user2, or user3)",
        required=True
    )
    
    args = parser.parse_args()
    user_id = args.user
    
    # Initialize database
    db = initialize_application()
    
    # Authenticate user
    authenticator = UserAuthenticator(db)
    
    if not authenticator.is_valid_user(user_id):
        print(f"Error: Invalid user ID '{user_id}'")
        print("Available users:")
        for uid, name in authenticator.list_available_users().items():
            print(f"  • {uid}: {name}")
        sys.exit(1)
    
    # Get user session data
    session_data = authenticator.get_user_session_data(user_id)
    if session_data is None:
        print(f"Error: Could not load session data for user {user_id}")
        sys.exit(1)
    
    # Display user information
    display_user_info(session_data)
    
    # Initialize chatbot
    try:
        chatbot = UniversityChatbot(user_id, db)
        print("Chatbot model loaded successfully.\n")
    except Exception as e:
        print(f"Error: Could not initialize chatbot model: {e}")
        print("Make sure Ollama is running with 'gemma3' and 'nomic-embed-text' models.")
        sys.exit(1)
    
    # Run chatbot
    run_chatbot(user_id, db, chatbot)
    print("\nThank you for using the University Chatbot!")


if __name__ == "__main__":
    main()
