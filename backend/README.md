# SATYA-V-2.0 Backend

## How to Run

1. From the project root, install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Start the backend server:
   ```bash
   uvicorn backend.main:app --reload
   ```

- Always run from the project root so imports work (utils, models, routes).
- API docs: http://localhost:8000/docs

## Structure
- All routes: `backend/routes/`
- Utilities: `backend/utils/`
- Models: `backend/models/`

## Best Practices
- Use virtual environments.
- Add new routes in `backend/routes/` and include them in `main.py`.
- Use robust error handling and logging (see route examples).
- For persistent scan history, integrate a database (see TODO). 