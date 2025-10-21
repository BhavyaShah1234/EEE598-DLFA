import config as c
import typing as t
import database as d

class UserAuthenticator:
    """Handle user authentication and session management."""
    def __init__(self, database: d.Database):
        self.db = database
        self.valid_users = list(c.SAMPLE_USERS.keys())

    def authenticate(self, user_id: str) -> t.Optional[t.Dict[str, t.Any]]:
        """
        Authenticate a user based on user_id.
        Returns user demographics if valid, None otherwise.
        """
        if user_id not in self.valid_users:
            return None

        user_data = self.db.get_user_demographics(user_id)
        if user_data is None:
            return None

        return user_data

    def is_valid_user(self, user_id: str) -> bool:
        """Check if a user_id is valid."""
        return user_id in self.valid_users

    def get_user_session_data(self, user_id: str) -> t.Optional[t.Dict[str, t.Any]]:
        """
        Get comprehensive session data for a user.
        Includes demographics, interests, and recent chat history.
        """
        if not self.is_valid_user(user_id):
            return None

        demographics = self.db.get_user_demographics(user_id)
        if demographics is None:
            return None
        chat_history = self.db.get_chat_history(user_id, limit=10)
        user_context = self.db.get_user_context(user_id)

        return {
            "user_id": user_id,
            "demographics": demographics,
            "chat_history": chat_history,
            "context": user_context
        }

    def list_available_users(self) -> t.Dict[str, str]:
        """List all available users with their names."""
        users = {}
        for user_id in self.valid_users:
            user_data = self.db.get_user_demographics(user_id)
            if user_data:
                users[user_id] = user_data.get('name', 'Unknown')
        return users
