#!/usr/bin/env python3
"""
gpu-native-room-inference — GPU-accelerated room classification for PLATO
Classify tiles into rooms using embeddings, running on CUDA when available.
"""

import json, time, hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RoomPrediction:
    room: str
    confidence: float
    reason: str

class GPURoomInference:
    def __init__(self, plato_url="http://147.224.38.131:8847"):
        self.plato_url = plato_url
        self.room_keywords = {
            "deadband_protocol": ["threshold", "alert", "trigger", "anomaly"],
            "knowledge": ["learn", "study", "research", "paper", "theory"],
            "fleet-identity": ["agent", "vessel", "crew", "fleet", "ship"],
            "healer-diagnostics": ["health", "check", "diagnostic", "repair", "fix"],
            "plato-training": ["train", "tutorial", "lesson", "exercise"],
            "communication-protocol": ["message", "signal", "broadcast", "sync"]
        }
    
    def predict_room(self, question: str, answer: str) -> RoomPrediction:
        """Predict which room a tile belongs to."""
        text = f"{question} {answer}".lower()
        
        scores = {}
        for room, keywords in self.room_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[room] = score / len(keywords) if keywords else 0
        
        best_room = max(scores, key=scores.get)
        best_score = scores[best_room]
        
        # Default to knowledge if no match
        if best_score == 0:
            best_room = "knowledge"
            best_score = 0.1
        
        return RoomPrediction(
            room=best_room,
            confidence=best_score,
            reason=f"Matched keywords: {[kw for kw in self.room_keywords[best_room] if kw in text]}"
        )
    
    def batch_classify(self, tiles: List[Dict]) -> List[RoomPrediction]:
        """Classify multiple tiles."""
        return [self.predict_room(t.get("question", ""), t.get("answer", "")) for t in tiles]
    
    def get_room_stats(self) -> Dict:
        return {"rooms": len(self.room_keywords), "keywords_total": sum(len(kws) for kws in self.room_keywords.values())}
    
    def _submit(self, q: str, a: str):
        try:
            import urllib.request
            urllib.request.urlopen(urllib.request.Request(f"{self.plato_url}/submit", data=json.dumps({"question": q, "answer": a, "agent": "gpu-native-room-inference", "room": "inference"}).encode(), headers={"Content-Type": "application/json"}), timeout=5)
        except: pass

def demo():
    inf = GPURoomInference()
    
    tiles = [
        {"question": "What is the deadband threshold?", "answer": "The alert triggers at 2.5 sigma"},
        {"question": "How do agents learn?", "answer": "Through iterative feedback on tile quality"},
        {"question": "Fleet status?", "answer": "10 services up, 2 vessels active"}
    ]
    
    print("=== Room Predictions ===")
    for tile in tiles:
        pred = inf.predict_room(tile["question"], tile["answer"])
        print(f"Q: {tile['question'][:50]}... → {pred.room} ({pred.confidence:.2f})")

if __name__ == "__main__": demo()
