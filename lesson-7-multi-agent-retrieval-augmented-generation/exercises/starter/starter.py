"""
Insurance Claims RAG System Exercise - Starter
==============================================

In this exercise, you'll enhance the insurance claims processing system by adding 
fraud detection capabilities powered by RAG. Fraud detection is a critical component 
of insurance claims processing, saving the industry billions of dollars annually.

Your task is to:

1. Implement a FraudDetectionAgent class that leverages RAG to identify potentially 
   fraudulent claims by comparing them with known fraud patterns
   
2. Create a fraud knowledge base with common fraud indicators and patterns
   
3. Implement vector search functionality to identify similar fraud patterns
   
4. Integrate the agent into the existing workflow, adding a fraud review step to the
   claim processing pipeline
"""

from typing import Dict, List, Any, Optional, Union, Set
import random
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)
import os
import dotenv

# Note: Make sure to set up your .env file with your API key before running
dotenv.load_dotenv(dotenv_path='../.env')
openai_api_key = os.getenv('UDACITY_OPENAI_API_KEY')

model = OpenAIServerModel(
    model_id='gpt-4o-mini',
    api_base='https://openai.vocareum.com/v1',
    api_key=openai_api_key,
)

# Import core components from the demo file
from demo.demo import (
    PrivacyLevel, AccessControl, Claim, PatientRecord, ComplaintRecord, 
    Database, VectorKnowledgeBase, VectorClaimSearch,
    DataGenerator, database, vector_kb, vector_claim_search,
    search_knowledge_base, retrieve_claim_history, get_claim_details,
    get_patient_info, find_similar_claims, submit_complaint,
    respond_to_complaint, get_complaint_history, process_new_claim,
    ClaimProcessingAgent, CustomerServiceAgent, MedicalReviewAgent
)

# STEP 1: Create a knowledge base of fraud patterns
fraud_knowledge_base = [
    {
        "pattern_id": "FP-001",
        "category": "Frequent Submissions",
        "description": "Patient files multiple high-value claims within a very short billing window across different providers.",
        "risk_score": 0.85
    },
    {
        "pattern_id": "FP-002",
        "category": "Inflated Billing",
        "description": "Billed amount significantly exceeds standard regional medical costs for the specified diagnostic procedure.",
        "risk_score": 0.90
    },
    {
        "pattern_id": "FP-003",
        "category": "Suspicious Diagnosis Match",
        "description": "Unusual combination of unrelated expensive elective treatments filed simultaneously by a single provider.",
        "risk_score": 0.75
    },
    {
        "pattern_id": "FP-004",
        "category": "Duplicate Services",
        "description": "Submitting identical billing codes and service timestamps for overlapping treatments.",
        "risk_score": 0.95
    }
]

# STEP 2: Implement a vector-based fraud pattern detector
class FraudPatternDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.patterns = []
        self.vectors = None
        self.update_patterns(fraud_knowledge_base)
        
    def update_patterns(self, fraud_patterns):
        self.patterns = fraud_patterns
        texts = [f"{p['category']}: {p['description']}" for p in self.patterns]
        if texts:
            self.vectors = self.vectorizer.fit_transform(texts)
        else:
            self.vectors = None
        
    def detect_fraud_indicators(self, claim: Claim, patient_history: List[Claim], access_level: str = PrivacyLevel.AGENT) -> Dict[str, Any]:
        # Rule-based and vector-based analysis
        indicators = []
        total_risk = 0.0
        
        # Check claim amount anomaly rule
        if claim.amount > 5000.0:
            indicators.append("High claim amount threshold exceeded (> $5,000).")
            total_risk += 0.3
            
        # Check frequency of recent claims in patient history
        if patient_history:
            recent_claims = [c for c in patient_history if (datetime.now() - datetime.fromisoformat(c.timestamp)).days < 30]
            if len(recent_claims) > 3:
                indicators.append(f"High frequency warning: {len(recent_claims)} claims filed within the last 30 days.")
                total_risk += 0.4
                
        # Vector similarity check against known fraud indicators
        claim_text = f"Diagnosis: {claim.diagnosis}, Description: {claim.description}, Amount: {claim.amount}"
        if self.vectors is not None:
            claim_vec = self.vectorizer.transform([claim_text])
            similarities = cosine_similarity(claim_vec, self.vectors).flatten()
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score > 0.25:
                matched_pattern = self.patterns[best_idx]
                indicators.append(f"Matches known fraud pattern ({matched_pattern['pattern_id']} - {matched_pattern['category']}): {matched_pattern['description']}")
                total_risk += matched_pattern['risk_score'] * best_score

        final_risk_score = min(1.0, total_risk)
        is_fraudulent = final_risk_score >= 0.6
        
        return {
            "claim_id": claim.claim_id,
            "risk_score": round(final_risk_score, 2),
            "is_suspected_fraud": is_fraudulent,
            "indicators": indicators
        }

# Global instance of fraud detector
fraud_detector = FraudPatternDetector()

# STEP 3: Implement a tool for fraud detection
@tool
def check_claim_for_fraud(claim_id: str, access_level: str = PrivacyLevel.AGENT) -> Dict:
    """
    Check a claim for potential fraud indicators.
    
    Args:
        claim_id: The claim ID to check
        access_level: The access level of the requester
        
    Returns:
        Dictionary containing fraud assessment results
    """
    claim = database.get_claim(claim_id)
    if not claim:
        return {"error": f"Claim {claim_id} not found."}
        
    patient_claims = database.get_patient_claims(claim.patient_id)
    assessment = fraud_detector.detect_fraud_indicators(claim, patient_claims, access_level)
    return assessment

# STEP 4: Create a FraudDetectionAgent
class FraudDetectionAgent(ToolCallingAgent):
    """Agent for detecting potential fraud in insurance claims."""
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[check_claim_for_fraud, get_claim_details, retrieve_claim_history],
            model=model,
            name="fraud_detection_agent",
            description="Agent responsible for reviewing insurance claims against known fraud patterns and evaluating risk scores."
        )

# STEP 5: Update the orchestrator to include fraud detection
class EnhancedClaimOrchestrator:
    """Orchestrator that integrates claim processing, medical review, and fraud detection."""
    def __init__(self, model):
        self.claim_agent = ClaimProcessingAgent(model)
        self.medical_agent = MedicalReviewAgent(model)
        self.fraud_agent = FraudDetectionAgent(model)
        
    def process_and_audit_claim(self, claim_id: str) -> str:
        """Processes a claim through standard workflow plus fraud inspection."""
        claim = database.get_claim(claim_id)
        if not claim:
            return f"Error: Claim {claim_id} does not exist."
            
        # Run Fraud Check via Agent
        fraud_result = self.fraud_agent.run(f"Evaluate claim {claim_id} for potential fraud indicators and return the risk score.")
        
        return f"Claim Audit Completed for {claim_id}.\nFraud Analysis Report:\n{fraud_result}"

# STEP 6: Function to demonstrate the fraud detection capabilities
def demonstrate_fraud_detection():
    """
    Run a demonstration of the fraud detection capabilities.
    """
    if not database.claims:
        print("No claims available in the database for demonstration.")
        return
        
    sample_claim = database.claims[0]
    print(f"Running Fraud Detection against sample claim: {sample_claim.claim_id}")
    
    fraud_agent = FraudDetectionAgent(model)
    evaluation = fraud_agent.run(f"Check claim {sample_claim.claim_id} for potential fraud and explain your findings.")
    
    print("\n--- Fraud Detection Evaluation Result ---")
    print(evaluation)

if __name__ == '__main__':
    # Initialize and populate database
    print('Initializing and populating database...')
    DataGenerator.populate_database(num_patients=20, num_claims=50, num_complaints=10)
    print(f"Database contains {len(database.patients)} patients, {len(database.claims)} claims, and {len(database.complaints)} complaints")
    
    # Run the fraud detection demo
    print('\n=== Insurance Claim Fraud Detection Demo ===\n')
    demonstrate_fraud_detection()