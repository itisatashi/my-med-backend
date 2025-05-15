from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import sqlite3
import logging
import uuid
from datetime import datetime, timedelta
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="MedAssyst API",
    description="AI Medical Assistant API using External Diagnosis Service",
    version="1.0.0"
)

# Configure CORS
# Add CORS middleware to allow frontend to communicate with this API
origins = [
    "https://my-med-frontend.vercel.app",  # Production Vercel frontend
    "http://localhost:5173",               # Local development server
    "http://localhost:5174",               # Alternative local port
    "*"                                   # Allow all origins (remove in strict production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],                  # Expose all headers
    max_age=600                           # Cache preflight requests for 10 minutes
)

# Database setup
def init_db():
    """Initialize the database with necessary tables"""
    conn = sqlite3.connect('medassyst.db')
    cursor = conn.cursor()
    
    # Create consultations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS consultations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symptoms TEXT NOT NULL,
        diagnosis TEXT NOT NULL,
        severity INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create a table for symptoms to better track frequency
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS symptoms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        consultation_id INTEGER,
        symptom_text TEXT NOT NULL,
        FOREIGN KEY (consultation_id) REFERENCES consultations (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print("Database initialized successfully!")

# Call init_db at startup
@app.on_event("startup")
def startup_event():
    init_db()

# Define input/output models
class SymptomRequest(BaseModel):
    symptoms: str

class ConsultationResponse(BaseModel):
    diagnosis: str
    consultation_id: int
    severity: Optional[int] = 0

# External API integration for medical diagnosis
def query_external_api(symptoms_text):
    """Query external medical API for diagnosis"""
    
    # External API endpoint
    EXTERNAL_API_URL = "https://begdulla.uz/APII/api.php"
    
    try:
        print(f"\n\n===== SENDING REQUEST TO EXTERNAL API =====\n")
        print(f"Sending request to external API at {EXTERNAL_API_URL}")
        print(f"Symptoms: {symptoms_text}")
        
        # Make request to external API
        response = requests.post(
            EXTERNAL_API_URL, 
            json={"prompt": symptoms_text},
            timeout=30  # 30 second timeout
        )
        
        # Print response details for debugging
        print(f"\n\n===== API RESPONSE =====\n")
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Parsed JSON successfully")
            
            # Get the response text
            model_response = result.get('response', 'Не удалось получить ответ от сервиса')
            print(f"\n\n===== API RESPONSE CONTENT =====\n{model_response[:300]}...\n\n") # Print first 300 chars
            
            return model_response
        else:
            print(f"API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"External API error: {response.text}"
            )
            
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to the external API. Please check your internet connection."
        )
    except requests.exceptions.Timeout as e:
        print(f"Timeout error: {str(e)}")
        raise HTTPException(
            status_code=504,
            detail="External API timeout. The service might be experiencing high load."
        )
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing API response: {str(e)}"
        )

# Helper function to get database connection
def get_connection():
    """Get a connection to the database"""
    return sqlite3.connect('medassyst.db')

# Save extracted symptoms for analytics
def save_symptoms(consultation_id, symptoms_text):
    """Extract and save individual symptoms from the text"""
    # Basic list of common symptoms in Russian
    common_symptoms = [
        "головная боль", "боль в голове", "тошнота", "рвота", "температура", 
        "слабость", "кашель", "насморк", "боль в горле", "боль в животе",
        "диарея", "сыпь", "зуд", "одышка", "усталость", "головокружение",
        "боль в груди", "боль в спине", "боль в суставах"
    ]
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Simple symptom extraction
    symptoms_text = symptoms_text.lower()
    for symptom in common_symptoms:
        if symptom in symptoms_text:
            cursor.execute(
                "INSERT INTO symptoms (consultation_id, symptom_text) VALUES (?, ?)",
                (consultation_id, symptom)
            )
    
    conn.commit()
    conn.close()

# Save consultation to database
def save_consultation(consultation_data: dict):
    """Save consultation to database"""
    logging.info(f"Saving consultation: {consultation_data}")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Calculate severity
        severity = consultation_data.get('severity', 0)
        
        # Insert into consultations table
        cursor.execute(
            "INSERT INTO consultations (symptoms, diagnosis, severity) VALUES (?, ?, ?)",
            (consultation_data['symptoms'], consultation_data['diagnosis'], severity)
        )
        
        # Get the ID of the new consultation
        consultation_id = cursor.lastrowid
        
        # Commit and close
        conn.commit()
        conn.close()
        
        print(f"Successfully saved consultation with ID: {consultation_id}")
        return consultation_id
    except Exception as e:
        print(f"ERROR saving consultation: {str(e)}")
        # Return a placeholder ID if saving fails to prevent further errors
        return -1

# Estimate the severity of symptoms
def estimate_severity(symptoms_text, diagnosis_text):
    """Estimate the severity of symptoms on a scale of 0-3"""
    # Basic implementation based on keyword matching
    severe_indicators = [
        "срочно", "немедленно", "экстренно", "опасно", "тяжелый", "критический",
        "боль в груди", "затрудненное дыхание", "потеря сознания"
    ]
    
    medium_indicators = [
        "высокая температура", "сильная боль", "рвота", "диарея", 
        "инфекция", "воспаление"
    ]
    
    severity = 0
    combined_text = (symptoms_text + " " + diagnosis_text).lower()
    
    for indicator in severe_indicators:
        if indicator in combined_text:
            return 3  # High severity
    
    for indicator in medium_indicators:
        if indicator in combined_text:
            severity = max(severity, 2)  # Medium severity
    
    # If symptoms are longer than 100 chars and severity is still 0, set to 1
    if len(symptoms_text) > 100 and severity == 0:
        severity = 1
        
    return severity

# Get the most common symptoms
def get_common_symptoms(limit=10):
    """Get the most common symptoms from the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT symptom_text, COUNT(*) as count 
    FROM symptoms 
    GROUP BY symptom_text 
    ORDER BY count DESC
    LIMIT ?
    """, (limit,))
    
    result = cursor.fetchall()
    conn.close()
    
    return [{"symptom": row[0], "count": row[1]} for row in result]

# API endpoints
@app.get("/")
@app.get("/health")
async def root():
    """Root endpoint for API health check"""
    return {"status": "MedAssyst API is running", "version": "1.0.0"}

@app.post("/api/consultation")
async def handle_consultation(request: Request):
    """Handle consultation request from the frontend's direct fetch API call"""    
    data = await request.json()
    symptoms = data.get('symptoms', '')
    
    if not symptoms:
        raise HTTPException(status_code=400, detail="No symptoms provided")
        
    try:
        return await process_consultation(symptoms)
    except Exception as e:
        logging.error(f"Error processing consultation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get consultation response: {str(e)}")

@app.post("/api/consult", response_model=ConsultationResponse)
async def get_consultation(request: SymptomRequest):
    """Get AI consultation response for symptoms using SymptomRequest model"""
    return await process_consultation(request.symptoms)
    
async def process_consultation(symptoms: str):
    """Get AI consultation response for symptoms"""
    try:
        print(f"Processing symptoms: {symptoms}")
        # Early input validation and cleaning
        if not symptoms or len(symptoms.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Symptoms description is too short. Please provide more details."
            )
            
        # Get diagnosis from external API
        try:
            diagnosis_text = query_external_api(symptoms)
        except HTTPException as e:
            raise e  # Re-raise API-specific exceptions
        except Exception as api_error:
            # Fallback to simple diagnosis if API fails
            print(f"Error querying external API: {api_error}")
            diagnosis_text = "Извините, не удалось подключиться к диагностическому сервису. "\
                             "Пожалуйста, проверьте подключение к интернету или попробуйте позже."
                             
        # Estimate severity
        severity = estimate_severity(symptoms, diagnosis_text)
        
        # Save consultation to database
        consultation_data = {
            "symptoms": symptoms,
            "diagnosis": diagnosis_text,
            "severity": severity
        }
        
        consultation_id = save_consultation(consultation_data)
        
        # Return response
        response = {
            "diagnosis": diagnosis_text,
            "consultation_id": consultation_id,
            "severity": severity
        }
        
        return response
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Error processing consultation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing consultation: {str(e)}"
        )

@app.get("/api/history")
async def get_history():
    """
    Get the consultation history from the database
    """
    try:
        # Use dictionary row factory
        conn = get_connection()
        conn.row_factory = lambda cursor, row: {
            col[0]: row[idx] for idx, col in enumerate(cursor.description)
        }
        cursor = conn.cursor()
        
        # Get all consultations with formatted dates
        cursor.execute(
            """SELECT 
                id, 
                symptoms,
                diagnosis,
                severity,
                datetime(created_at, 'localtime') as created_at
            FROM consultations 
            ORDER BY created_at DESC"""
        )
        
        history = cursor.fetchall()
        conn.close()
        
        # Format the data for better frontend display
        for item in history:
            # Limit symptoms length for display
            if item.get('symptoms') and len(item.get('symptoms', '')) > 100:
                item['symptoms_preview'] = item['symptoms'][:100] + '...'
            else:
                item['symptoms_preview'] = item.get('symptoms', '')
                
        print(f"Returning {len(history)} history items")
        return history  # Return directly as array for easier frontend handling
    except Exception as e:
        print(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch consultation history: {str(e)}")

# Delete a specific consultation by ID
@app.delete("/api/history/{consultation_id}")
async def delete_consultation(consultation_id: int):
    """
    Delete a specific consultation by ID
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Delete the consultation record
        cursor.execute("DELETE FROM consultations WHERE id = ?", (consultation_id,))
        
        # Delete associated symptoms
        cursor.execute("DELETE FROM symptoms WHERE consultation_id = ?", (consultation_id,))
        
        # Commit the changes
        conn.commit()
        conn.close()
        
        return {"message": f"Consultation {consultation_id} deleted successfully", "status": "success"}
    except Exception as e:
        print(f"Error deleting consultation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete consultation: {str(e)}")

@app.get("/api/analytics")
async def get_analytics():
    """
    Get analytics data from the consultations
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get count of consultations by day
    cursor.execute("""
    SELECT date(created_at) as day, COUNT(*) as count 
    FROM consultations 
    GROUP BY date(created_at)
    ORDER BY day DESC
    LIMIT 30
    """)
    daily_counts = [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]
    
    # Get count by week
    cursor.execute("""
    SELECT strftime('%Y-%W', created_at) as week, COUNT(*) as count 
    FROM consultations 
    GROUP BY week
    ORDER BY week DESC
    LIMIT 12
    """)
    weekly_counts = [{"week": f"Week {row[0].split('-')[1]} ({row[0].split('-')[0]})", "count": row[1]} for row in cursor.fetchall()]
    
    # Get common symptoms
    top_symptoms = get_common_symptoms(10)
    
    # Extract common diagnoses by keyword
    cursor.execute("""
    SELECT diagnosis FROM consultations
    ORDER BY created_at DESC
    LIMIT 100
    """)
    diagnoses = cursor.fetchall()
    
    # Create a dictionary to count diagnosis types
    diagnosis_counts = {}
    for row in diagnoses:
        diagnosis_text = row[0].lower() if row[0] else ""
        
        # Check for common medical condition keywords
        categories = {
            "Инфекция": ["инфекция", "инфекционн", "бактери", "вирус"],
            "Воспаление": ["воспал", "отек", "отёк", "покрасн"],
            "Травма": ["травм", "ушиб", "перелом", "растяжение", "вывих"],
            "Аллергия": ["аллерг", "сыпь", "зуд", "крапивниц"],
            "Хроническое": ["хронич", "длительн", "постоянн"],
            "Кариес": ["кариес", "пульпит", "зубной"],
            "Головные боли": ["головн", "мигрен", "боль в голове"],
            "Респираторное": ["простуд", "грипп", "бронхит", "пневмония", "орви", "орз", "насморк"],
            "Кожное": ["кожн", "дерматит", "экзем"],
            "Пищеварительное": ["желудоч", "кишечн", "гастрит", "изжог"]
        }
        
        # Count the matches in each category
        categorized = False
        for category, keywords in categories.items():
            if any(keyword in diagnosis_text for keyword in keywords):
                diagnosis_counts[category] = diagnosis_counts.get(category, 0) + 1
                categorized = True
        
        # Count those that didn't match any category
        if not categorized:
            diagnosis_counts["Другое"] = diagnosis_counts.get("Другое", 0) + 1
    
    # Convert to list format for the frontend
    diagnosis_distribution = [{"diagnosis": k, "count": v} for k, v in diagnosis_counts.items() if v > 0]
    # Sort by count descending
    diagnosis_distribution.sort(key=lambda x: x["count"], reverse=True)
    
    # Get severity distribution
    cursor.execute("""
    SELECT severity, COUNT(*) as count
    FROM consultations
    GROUP BY severity
    ORDER BY severity
    """)
    severity_data = cursor.fetchall()
    
    severity_labels = {
        1: "Не срочно",
        2: "Требует внимания",
        3: "Срочно",
        0: "Не определено"
    }
    
    severity_distribution = [
        {"severity": severity_labels.get(row[0], "Не определено"), "count": row[1]}
        for row in severity_data
    ]
    
    conn.close()
    
    return {
        "daily_counts": daily_counts,
        "weekly_counts": weekly_counts,
        "top_symptoms": top_symptoms,
        "diagnosis_distribution": diagnosis_distribution,
        "severity_distribution": severity_distribution
    }

@app.get("/api/healthcheck")
@app.get("/api/health")
async def check_api_health():
    """
    Check if the external API is available
    """
    try:
        # Check external API health
        EXTERNAL_API_URL = "https://begdulla.uz/APII/api.php"
        response = requests.get(EXTERNAL_API_URL, timeout=5)
        
        if response.status_code == 200:
            return {
                "status": "online",
                "message": "External API is available"
            }
        else:
            return {
                "status": "degraded",
                "message": f"External API returned status code {response.status_code}"
            }
    except Exception as e:
        print(f"Error checking API health: {str(e)}")
        return {
            "status": "offline",
            "message": "External API is not available"
        }

# API endpoint to proxy requests to the external AI API
@app.post("/api/proxy")
async def proxy_ai_request(request: Request):
    """
    Proxy requests to the external AI API to avoid CORS issues
    """
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        
        # Use the existing query_external_api function
        result = query_external_api(prompt)
        
        # Save this consultation to history automatically
        # Estimate severity based on text analysis
        severity = estimate_severity(prompt, result)
        
        # Save consultation to database
        consultation_id = save_consultation({
            'symptoms': prompt,
            'diagnosis': result,
            'severity': severity
        })
        
        # Return both the AI response and the consultation ID
        return {
            "response": result,
            "consultation_id": consultation_id,
            "severity": severity
        }
    except Exception as e:
        print(f"Error in proxy endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling external API: {str(e)}")

# Endpoint to delete all consultation history (use with caution)
@app.delete("/api/history/all")
async def delete_all_history():
    """
    Delete all consultation history from the database
    WARNING: This will permanently delete all consultation records
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Delete all records from consultations table
        cursor.execute("DELETE FROM consultations")
        
        # Delete all records from symptoms table
        cursor.execute("DELETE FROM symptoms")
        
        # Commit the changes
        conn.commit()
        conn.close()
        
        return {"message": "All consultation history has been deleted successfully", "status": "success"}
    except Exception as e:
        print(f"Error deleting history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete consultation history: {str(e)}")

# Run the FastAPI app with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
