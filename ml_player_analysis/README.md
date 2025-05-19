# ML-Powered Game Analytics – Memory Card Game 🎮📊

## 🔍 Project Overview

This project adds machine learning-powered analytics to a simple Memory Card Game. It tracks how players interact with the game (e.g., number of flips, mismatches, duration) and applies clustering and prediction models to understand player behavior and suggest game design improvements.

## 🧠 Technologies Used

- **Game**: HTML, CSS, JavaScript
- **ML & Analytics**: Python (pandas, scikit-learn, matplotlib, seaborn)
- **Visualization**: Matplotlib plots for clusters and difficulty prediction
- **Tools**: GitHub for version control, Jupyter or VS Code for scripting

## 📂 File Structure


## 📊 Machine Learning Features

### ✅ K-Means Clustering
- Groups players based on flip count, mismatches, and session time
- Reveals different player types (e.g., slow vs. fast, careful vs. error-prone)

### ✅ Decision Tree Prediction
- Predicts whether a player will struggle based on early-game behavior
- Labels data with a custom "struggled" score
- Accuracy from model: **100% (on small dataset)**

## 💡 Design Insights (Automatically Generated)

- Players who struggled flipped **46 cards on average** (vs **37.8 overall**)
- **Cluster 2** had the highest mismatches — these players may be confused
- Sessions longer than **35 seconds** were flagged as potentially frustrating

**Recommendations:**
- Add a **hint button** for players who flip > 40 cards
- Improve card contrast to reduce mismatches
- Consider adding a **progress bar or pacing indicator**

## ▶️ How to Run It

1. Install required packages:

2. Make sure all JSON files are in `player data/`

3. Run the ML analysis:

4. Two charts will display:
- Player clusters
- Difficulty prediction (red = struggled)

5. Insights print at the end

## 🎬 Video Demo

_Link to your video goes here (YouTube or Google Drive)_


