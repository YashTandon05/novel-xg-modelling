# Beyond Distance and Angle: A Comparative Study of Shot Quality Models in Soccer

A novel expected goals (xG) model built on StatsBomb open data, extending standard geometry-based xG with goalkeeper position and freeze-frame features.

## Live Demo

**[Launch the demo →](https://yashtandon05.github.io/novel-xg-modelling)**

> ⚠️ **Note:** The prediction API is hosted on a free Hugging Face Space, which sleeps after inactivity. If your first prediction takes 30–60 seconds to respond, just wait — it's waking up. Subsequent predictions will be instant.

---

## How to use the demo

The demo shows the **attacking half** of the pitch. The goal mouth is on the right edge.

1. **Place the shot** — select *Shot* and click anywhere on the pitch to set where the ball is struck from
2. **Place the goalkeeper** — select *Goalkeeper* and click to position the keeper between the shot and goal
3. **Add defenders** — select *Defender* and click to add outfield players blocking the shot; you can add as many as you like
4. **Choose a body part** — select *Right Foot*, *Left Foot*, or *Head* from the dropdown
5. **Click Predict** — the model returns the xG as a percentage

To start over, click **Clear**.

### Tips
- Move the shot closer to goal and watch the xG rise
- Place the GK off-centre and see how it affects the probability
- Add defenders between the shot and goal to reduce the available angle
- Headers from close range have lower xG than you might expect — heading is heavily penalised by the model

---

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn app:app --reload

# Serve the UI
python -m http.server 8080 --directory docs
```

Then open [http://localhost:8080](http://localhost:8080).
