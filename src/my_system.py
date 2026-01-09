import time
import random

class MyEdgeSystem:
    def __init__(self, config_name="base"):
        self.config = config_name
        print(f"System Initialized: {config_name}")

    def predict(self, query: str):
        """
        Simulates your system's response.
        Returns a standardised dictionary needed by the evaluators.
        """
        # --- START: Replace this block with your REAL System Logic ---
        start_time = time.time()
        
        # Simulated logic
        answer = "July 4, 2026 is a Saturday."
        retrieved_docs = ["Independence Day in USA is on July 4.", "2026 calendar data."]
        tools_used = ["Calendar_Tool"]
        
        # Simulate processing time
        time.sleep(0.1) 
        # --- END ---

        latency = time.time() - start_time
        
        # Return strict format
        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "tools_used": tools_used,
            "latency": latency,
            "query": query # Useful to pass back
        }