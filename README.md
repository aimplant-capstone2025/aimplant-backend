# AImplant Backend

## Overview

This is the backend service for the **AImplant** project, built with **FastAPI** to provide API endpoints for handling data processing and AI-based tasks.

### Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### Setup

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/aimplant-capstone2025/aimplant-backend.git
cd aimplant-backend
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

#### Windows
```bash
.\venv\Scripts\Activate
```
#### macOS/Linux
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Running the Backend Server

Start the development server with:

```bash
uvicorn main:app --reload
```

For production, use **Gunicorn**:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```


