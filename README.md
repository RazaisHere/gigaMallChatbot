# Mall Chatbot Backend

## Setup Instructions

### Phase 0: Environment Setup

#### 1. Install PostgreSQL

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

**Windows:**
Download from [postgresql.org](https://www.postgresql.org/download/windows/)

#### 2. Create Database

```bash
# Login to PostgreSQL
sudo -u postgres psql  # Linux/macOS
# or on Windows, use pgAdmin or psql from command line

# In PostgreSQL shell:
CREATE DATABASE mall_chatbot;
CREATE USER mall_admin WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE mall_chatbot TO mall_admin;
\q
```

#### 3. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

#### 4. Configure Environment Variables

Create a `.env` file in the `backend` directory:

```env
DATABASE_URL=postgresql://mall_admin:your_secure_password@localhost/mall_chatbot
GOOGLE_API_KEY=your_gemini_api_key_here
```

**Note:** Replace `your_secure_password` with your actual PostgreSQL password and `your_gemini_api_key_here` with your Google Gemini API key.

#### 5. Initialize Database

```bash
python init_db.py
```

This will create the necessary tables and populate them with sample data (stores and mall information).

### Running the Server

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API Documentation (Swagger UI): `http://localhost:8000/docs`

## API Endpoints

### `POST /chat`
Main chat endpoint with conversation history.

**Request:**
```json
{
  "message": "Hello!",
  "session_id": "optional-session-id",
  "user_id": "anonymous"
}
```

**Response:**
```json
{
  "response": "Hello! How can I help you today?",
  "session_id": "generated-or-provided-session-id"
}
```

### `GET /history/{session_id}`
Get conversation history for a session.

**Response:**
```json
{
  "session_id": "session-id",
  "messages": [
    {
      "role": "user",
      "message": "Hello!",
      "timestamp": "2024-01-01T12:00:00"
    },
    {
      "role": "assistant",
      "message": "Hello! How can I help you?",
      "timestamp": "2024-01-01T12:00:01"
    }
  ]
}
```

### `DELETE /history/{session_id}`
Clear conversation history for a session.

### `GET /stores`
List all stores in the mall.

**Response:**
```json
{
  "stores": [
    {
      "name": "Zara",
      "floor": "2",
      "location": "Near Food Court, Section A",
      "category": "Fashion"
    }
  ]
}
```

## Project Structure

```
backend/
├── main.py              # FastAPI application
├── database.py          # SQLAlchemy models and database setup
├── chat_history.py      # Conversation history management
├── config.py            # Configuration settings
├── init_db.py           # Database initialization script
├── services/
│   └── llm.py          # LLM service (not used in new approach)
└── requirements.txt     # Python dependencies
```

## Success Criteria

✅ Can send messages and get responses
✅ Conversation history is saved in PostgreSQL
✅ Can view history via `/history/{session_id}`
✅ Follow-up questions work (uses context)
✅ Can list stores via `/stores` endpoint

## Testing

Test the API using the Swagger UI at `http://localhost:8000/docs` or use curl:

```bash
# Send a message
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Get history
curl "http://localhost:8000/history/{session_id}"

# List stores
curl "http://localhost:8000/stores"
```

## Next Steps

After Phase 0 and Phase 1 are complete, you can add:
- Intent detection integration
- Store-specific queries
- Mall hours queries
- Parking information queries
- Enhanced context handling
