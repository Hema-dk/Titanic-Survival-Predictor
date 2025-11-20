<div align="center">
  <h1>🛳️ Titanic Survival Prediction — Logistic Regression</h1>
  <p><em>Data cleaning · EDA · Feature engineering · Model training · Evaluation</em></p>


  <p>
    <span style="background:#eef2ff; padding:6px 12px; border-radius:8px; margin:4px;">Python</span>
    <span style="background:#eef2ff; padding:6px 12px; border-radius:8px; margin:4px;">scikit-learn</span>
    <span style="background:#eef2ff; padding:6px 12px; border-radius:8px; margin:4px;">pandas</span>
    <span style="background:#eef2ff; padding:6px 12px; border-radius:8px; margin:4px;">seaborn</span>
  </p>
</div>

<hr/>

<h2> Project Overview</h2>
<p>
This project builds a Machine Learning model to predict whether a passenger survived the Titanic disaster.
It includes data cleaning, exploratory analysis, feature engineering, model training, and evaluation.
</p>

<ul>
  <li>Binary classification problem</li>
  <li>Target variable: <strong>Survived</strong> (1 = Yes, 0 = No)</li>
  <li>Dataset: <strong>titanic_train.csv</strong></li>
</ul>

<hr/>

<h2> Dataset</h2>
<p><strong>Rows:</strong> 891 passengers<br>
<strong>Contains:</strong> Age, Gender, Ticket info, Fare, Family size, Survival</p>

<hr/>

<h2> Data Cleaning</h2>
<ul>
  <li>Imputed missing <strong>Age</strong> using median based on Pclass</li>
  <li>Dropped <strong>Cabin</strong> (85% missing)</li>
  <li>Removed <strong>Name</strong> and <strong>Ticket</strong></li>
  <li>Encoded <strong>Sex</strong> and <strong>Embarked</strong> using one-hot encoding</li>
</ul>

<hr/>

<h2> Exploratory Data Analysis</h2>
<ul>
  <li>Heatmap of missing values</li>
  <li>Survival count distribution</li>
  <li>Survival by Sex</li>
  <li>Survival by Pclass</li>
  <li>Age and Fare histogram</li>
  <li>Boxplot of Age vs Pclass</li>
</ul>

<hr/>

<h2> Feature Engineering</h2>
<ul>
  <li>Created dummy variables with <code>get_dummies()</code></li>
  <li>Scaled features using <code>StandardScaler()</code></li>
</ul>

<hr/>

<h2> Model Training</h2>

<div style="background:#1b1b1b; color:white; padding:15px; border-radius:8px; font-family:Consolas, monospace; font-size:14px;">
<pre style="white-space:pre; margin:0;">
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
</pre>
</div>

<hr/>

<h2> Model Evaluation</h2>

<div style="background:#1b1b1b; color:white; padding:15px; border-radius:8px; font-family:Consolas, monospace; font-size:14px;">
<pre style="white-space:pre; margin:0;">
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))
</pre>
</div>

<hr/>

<h2> How to Run This Project</h2>

<h3>1️ Clone the repository</h3>
<div style="background:#1b1b1b; color:white; padding:15px; border-radius:8px;">
<pre>git clone https://github.com/your-username/your-repo.git
cd your-repo</pre>
</div>

<h3>2️ Install dependencies</h3>
<div style="background:#1b1b1b; color:white; padding:15px; border-radius:8px;">
<pre>pip install -r requirements.txt</pre>
</div>

<h3>3️ Add the dataset</h3>
<p>Place <strong>titanic_train.csv</strong> inside the project folder.</p>

<h3>4️ Run the Jupyter Notebook</h3>
<div style="background:#1b1b1b; color:white; padding:15px; border-radius:8px;">
<pre>jupyter notebook</pre>
</div>

<hr/>

<h2> Tech Stack</h2>
<ul>
  <li>Python</li>
  <li>NumPy</li>
  <li>pandas</li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
  <li>Scikit-learn</li>
</ul>

<hr/>

<h2> Future Improvements</h2>
<ul>
  <li>Try Random Forest, XGBoost, SVM</li>
  <li>Hyperparameter tuning (GridSearch / RandomizedSearch)</li>
  <li>Cross-validation</li>
  <li>Feature importance (SHAP)</li>
  <li>Deploy using Streamlit or Flask</li>
</ul>

<hr/>


