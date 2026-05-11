curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "how is my portfolio",
    "session_id": "v",
    "user_context": {
      "user_id": "u1",
      "positions": [
        {"ticker": "AAPL", "quantity": 10, "avg_cost": 150, "currency": "USD"}
      ]
    }
  }'