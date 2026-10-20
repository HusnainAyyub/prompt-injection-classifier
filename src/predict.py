import sys, joblib, os
if not os.path.exists("model.joblib"):
    print("Train first: python src/train.py"); raise SystemExit(1)
model = joblib.load("model.joblib")
prompt = " ".join(sys.argv[1:]) or "What is the capital of Pakistan?"
print(model.predict([prompt])[0])
