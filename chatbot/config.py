# config.py
import pathlib as p

# Database Configuration
DATABASE_DIR = p.Path("./data")
DATABASE_DIR.mkdir(exist_ok=True)
SQLITE_DB_PATH = DATABASE_DIR / "chatbot.db"
VECTOR_DB_PATH = DATABASE_DIR / "vector_db"
CHAT_HISTORY_DB = DATABASE_DIR / "chat_history"

# LLM Configuration
LLM_MODEL = "gemma3"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# LangGraph Configuration
GRAPH_CHECKPOINT_DIR = DATABASE_DIR / "graph_checkpoints"
GRAPH_CHECKPOINT_DIR.mkdir(exist_ok=True)

# Application Configuration
MAX_CHAT_HISTORY = 20
VECTOR_DB_CHUNK_SIZE = 500
VECTOR_DB_OVERLAP = 50

# Sample users
SAMPLE_USERS = {
    "user1": {
        "name": "Alice Johnson",
        "age": 20,
        "gender": "Female",
        "ethnicity": "Caucasian",
        "hometown": "Boston, MA",
        "education": "Computer Science, Year 2",
        "email": "alice@university.edu",
        "password": "qwertyuiop",
    },
    "user2": {
        "name": "Raj Patel",
        "age": 19,
        "gender": "Male",
        "ethnicity": "Indian",
        "hometown": "San Francisco, CA",
        "education": "Engineering, Year 1",
        "email": "raj@university.edu",
        "password": "qwertyuiop",
    },
    "user3": {
        "name": "Maria Garcia",
        "age": 21,
        "gender": "Female",
        "ethnicity": "Hispanic",
        "hometown": "Miami, FL",
        "education": "Business Administration, Year 3",
        "email": "maria@university.edu",
        "password": "qwertyuiop",
    }
}

# Sample university information
SAMPLE_UNIVERSITY_DATA = [
    {
        "title": "Computer Science Department",
        "content": "The Computer Science Department offers undergraduate and graduate programs. We have state-of-the-art labs, experienced faculty, and opportunities for internships at top tech companies.",
        "page": "departments/cs"
    },
    {
        "title": "Student Housing",
        "content": "The university offers on-campus housing for freshmen and sophomores. Housing includes dorms, apartments, and special interest housing. Apply through the housing portal by March 15.",
        "page": "housing"
    },
    {
        "title": "Library Resources",
        "content": "The main library offers 24/7 access to computers, study spaces, and thousands of physical and digital resources. Special collections include rare books and archives.",
        "page": "library"
    },
    {
        "title": "Career Services",
        "content": "Career Services helps students with resume writing, job interviews, and internship placement. Schedule a consultation with a career counselor free of charge.",
        "page": "career_services"
    },
    {
        "title": "Student Health Services",
        "content": "The Student Health Center provides medical care, mental health counseling, and wellness programs. Services are included in student fees.",
        "page": "health"
    },
    {
        "title": "Dining Services",
        "content": "Multiple dining halls and cafes across campus serve breakfast, lunch, and dinner. Meal plans are required for freshmen. Vegetarian, vegan, and allergen-free options available.",
        "page": "dining"
    },
    {
        "title": "Athletics and Recreation",
        "content": "The university has NCAA Division I sports teams and recreational facilities including gym, pool, basketball courts, and outdoor sports areas. Open to all students.",
        "page": "athletics"
    },
    {
        "title": "Tuition and Financial Aid",
        "content": "Annual tuition is $45,000. Financial aid packages include grants, loans, and work-study. Complete FAFSA to be considered for aid.",
        "page": "financial_aid"
    },
    {
        "title": "Scholarships",
        "content": "Merit-based scholarships up to $20,000 per year are available. Application deadlines vary by scholarship. Awards based on academics, athletics, or community service.",
        "page": "scholarships"
    },
    {
        "title": "Study Abroad Programs",
        "content": "Exchange programs in 50+ countries for one semester or full year. Scholarships available. Deadlines are semester-based.",
        "page": "study_abroad"
    }
]
