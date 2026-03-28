import config
import real_messaging
import real_ai_engine

print("--- TESTING TWILIO SMS ---")
try:
    res = real_messaging.send_sms('+917357138972', 'PDIE Auth Test')
    print("Result:", res)
except Exception as e:
    print("Error:", e)

print("\n--- TESTING GEMINI AI ---")
try:
    res = real_ai_engine.generate_response("Say 'Hello Judges' in 3 words.")
    print("Result:", res)
except Exception as e:
    print("Error:", e)
