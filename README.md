# Project Submission Module — Milestone 1 (Week 1-2)

## What this is
- `frontend/index.html` — nav bar ("Project Input") + Project Submission form
  (name, industry/sector, business model, target market, budget, description).
- `backend/main.py` — FastAPI service that validates and stores submissions.
- `backend/schema.sql` — PostgreSQL schema (`users`, `projects` tables).

## Setup

1. **Create the database and tables**
   ```bash
   createdb failure_prediction_ai
   psql -d failure_prediction_ai -f backend/schema.sql
   ```

2. **Set DB credentials** (env vars, or edit `DB_CONFIG` in `main.py` directly)
   ```bash
   export DB_HOST=localhost
   export DB_NAME=failure_prediction_ai
   export DB_USER=postgres
   export DB_PASSWORD=yourpassword
   ```

3. **Run the backend**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```

4. **Open the frontend**
   Just open `frontend/index.html` directly in a browser (double-click, or
   `python3 -m http.server 5500` from the `frontend/` folder and visit
   `http://localhost:5500`).

5. **Verify storage**
   ```
   GET http://localhost:8000/api/projects
   ```
   should return everything you've submitted.

## Design notes (for the mentor review)
- `industry_sector`, `business_model`, `target_market`, `budget`, `description`
  map directly onto the `projects` table from the ERD; `business_model` is new
  (added for today's form) and can be folded into `project_type` later if the
  schema gets consolidated — flag this with your mentor rather than silently
  diverging from the shared ERD.
- Validation lives in the Pydantic model (`ProjectSubmission`), not just the
  frontend — so bad data can't reach PostgreSQL even if someone hits the API
  directly (Postman, curl, etc.).
- No `user_id` / auth wiring yet — out of scope for Week 1-2 per the milestone
  breakdown. `user_id` in `projects` is nullable for now; wire it up once the
  Authentication Layer module starts.
