# Smart Failure Detection System

An AI-powered web application that helps evaluate business projects by analyzing project details and generating intelligent failure risk assessments, SWOT analysis, business health evaluation, and actionable recommendations.

---

## Project Overview

Smart Failure Detection is designed to assist entrepreneurs, startups, and business analysts in evaluating business ideas by identifying potential risks and providing strategic insights before implementation.

---

## Features

- Project Submission Form
- AI-Based Project Analysis
- Failure Risk Assessment
- Success Probability Prediction
- Business Health Evaluation
- SWOT Analysis
- Risk Breakdown
- Milestone Timeline
- Recommendations for Risk Reduction
- PostgreSQL Database Integration
- Responsive Web Interface

---

## Tech Stack

### Frontend

- HTML5
- CSS3
- JavaScript

### Backend

- Python
- FastAPI

### Database

- PostgreSQL (Render)

### Deployment

- Render
- GitHub

---

## Project Structure

```text
Smart-Failure-Detection/
│
├── index.html
├── analysis-results.html
├── main.py
├── schema.sql
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/pradeep5126/Smart-Failure-Detection.git
```

### Navigate to the Project

```bash
cd Smart-Failure-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
uvicorn main:app --reload
```

The application will be available at:

```
http://127.0.0.1:8000
```

---

## Live Demo

Application:

https://smart-failure-detection-ii27.onrender.com/

GitHub Repository:

https://github.com/pradeep5126/Smart-Failure-Detection

---

## Application Workflow

1. Open the application.
2. Fill in the project details.
3. Submit the form.
4. Project information is stored in PostgreSQL.
5. The system analyzes the submitted project.
6. View the generated:
   - Failure Risk
   - Success Probability
   - Business Health
   - SWOT Analysis
   - Risk Breakdown
   - Recommendations
   - Milestone Timeline

---

## Database

The application uses PostgreSQL hosted on Render to securely store submitted project information.

---

## Future Enhancements

- User Authentication
- Dashboard for Multiple Projects
- PDF Report Generation
- Email Notifications
- Enhanced AI-Based Analysis
- Analytics Dashboard

---