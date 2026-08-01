-- Matches the "Projects" table from your ERD (project_id, user_id, project_name,
-- project_type, budget, target_market, description, created_at).
-- industry/sector -> project_type, business_model -> stored in description-adjacent column.
-- Extended slightly to fit today's form fields exactly.

CREATE TABLE IF NOT EXISTS users (
    user_id     SERIAL PRIMARY KEY,
    name        VARCHAR(150) NOT NULL,
    email       VARCHAR(150) UNIQUE NOT NULL,
    role        VARCHAR(50) DEFAULT 'founder',
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS projects (
    project_id      SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(user_id),
    project_name    VARCHAR(200) NOT NULL,
    industry_sector VARCHAR(150) NOT NULL,
    business_model  VARCHAR(150) NOT NULL,
    target_market   VARCHAR(200) NOT NULL,
    budget          NUMERIC(14,2) NOT NULL,
    description     TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW()
);
