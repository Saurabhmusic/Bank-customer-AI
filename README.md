# Bank Marketing Campaign: Prediction & Customer Segmentation

###  **Live App:** [Click Here to Predict Term Deposit](YAHAN_APNA_STREAMLIT_APP_LINK_DALNA)

---

##  About The Project
In the banking industry, telemarketing campaigns are often hit-or-miss. Calling every customer is not only expensive but also annoying for those who aren't interested. 

I built this project to answer two key questions for the marketing team:
1.  **Who is most likely to say "Yes"?** (Prediction)
2.  **Who are our customers really?** (Segmentation)

By analyzing past campaign data, I built a machine learning solution that helps target the right people, saving time and resources.

---

##  My Approach
Instead of just fitting a model, I broke the problem down into logical steps to ensure the solution is robust and actionable.

### 1. Data Cleaning & Leakage Prevention
The raw dataset had a column `duration` (call duration). While highly correlated with the target, **I decided to drop it**. 
* **Reason:** We don't know the call duration *before* making the call. Including it would cause **Data Leakage** and give unrealistic accuracy. I wanted a model that works in real-world scenarios.

### 2. Analyzing Customer Behavior (EDA)
I dug into the data to find patterns. Some interesting observations:
* **Seasonality:** May is the busiest month for calls, but March and September have higher conversion rates.
* **Demographics:** Retired individuals and students are more likely to subscribe than blue-collar workers.

### 3. Customer Segmentation (Unsupervised Learning)
I didn't want to treat all customers the same. Using **K-Means Clustering**, I grouped customers into 4 distinct segments based on their Age, Balance, and Campaign interaction:
* **The "Whales":** Older customers with very high balances (High potential for investment products).
* **The "Calculated Savers":** Middle-aged, average balance, careful with spending.
* **The "Young Starters":** Low balance, mostly students or early career professionals.
* **The "Frequent Targets":** People who were called too many times (likely annoyed).

### 4. Handling Imbalanced Data
The dataset was heavily skewed — only **11%** of customers subscribed.
* **Challenge:** A standard model would just predict "No" for everyone and still get 89% accuracy.
* **Solution:** I used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data, ensuring the model learns to identify the "Yes" cases effectively.

### 5. Model Building & Deployment
I chose **Random Forest** because it handles mixed data types well and offers feature importance.
* **Performance:** Achieved an **F1-Score of ~0.90** and an **ROC-AUC of ~0.78**.
* **Deployment:** Built an interactive **Streamlit App** so non-technical stakeholders (like marketing managers) can use the model easily.

---

##  Tech Stack
* **Python** (Pandas, NumPy)
* **Machine Learning:** Scikit-Learn (Random Forest, K-Means, PCA)
* **Imbalance Handling:** SMOTE
* **Visualization:** Matplotlib, Seaborn
* **App Framework:** Streamlit

---

##  How to Run This Project
If you want to run this on your local machine:

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Bank-Customer-AI.git](https://github.com/YOUR_USERNAME/Bank-Customer-AI.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---

##  Future Improvements
If I had more time, I would:
* Test **XGBoost** or **LightGBM** to see if they outperform Random Forest.
* Deploy the model using **Docker** or as a **REST API** (FastAPI) for better integration.
* Add more financial indicators (like interest rates) if external data were available.

---
*Thanks for checking out my project! Feel free to reach out if you have feedback.*
