# Big Data Analytics - Assignment 2 (160122771047)

## Environment
- **Platform:** Google Colab
- **Language:** Python 3
- **Framework:** Apache Spark (via PySpark)
- **Setup:**
  - Java JDK 11
  - Spark 3.4.1
  - `pyspark`, `findspark`

---

## Project Structure
```
assignment/
│
├── 1_classifier_with_spark.ipynb          # Spark-based classification model
├── 2_clustering_with_spark.ipynb.         # Spark-based clustering model
└── 3_recommender_sys_with_spark.ipynb     # Spark-based movie recommendation system
```

---

## 1⃣ Classification Model

### Objective:
Build a classification model using Apache Spark to predict a target class based on input features.

### Dataset:
Used the **Iris dataset**, a classic multiclass classification problem with features like petal/sepal length & width.

### Model:
- **Algorithm:** Logistic Regression
- **Pipeline:** StringIndexer → VectorAssembler → LogisticRegression
- **Evaluation:** Accuracy, Confusion Matrix

### Result:
Achieved high accuracy (~95%+) on test data.

---

## 2⃣ Clustering Model

### Objective:
Perform unsupervised clustering on a dataset using Spark’s MLlib.

### Dataset:
Used the **Iris dataset** again, but this time without labels.

### Model:
- **Algorithm:** KMeans Clustering
- **Pipeline:** VectorAssembler → KMeans
- **Evaluation:** Silhouette Score

### Result:
The clustering grouped the dataset into 3 distinct clusters (aligned well with actual classes).

---

## 3⃣ Recommendation Engine

### Objective:
Create a collaborative filtering recommendation system to suggest movies to users.

### Dataset:
**MovieLens 100k** dataset (user ratings for movies)

### Model:
- **Algorithm:** Alternating Least Squares (ALS)
- **Process:**
  - Data cleaning and transformation
  - Training ALS model
  - Generating recommendations for users

### Result:
Successfully generated top-N movie recommendations for individual users.

---

## Setup Instructions (for all notebooks)
Paste and run the following in a Colab cell to install Spark:
```python
!apt-get install openjdk-11-jdk -y
!wget -q https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz
!tar -xzf spark-3.4.1-bin-hadoop3.tgz
!pip install -q pyspark
```

Then initialize Spark:
```python
import os
from pyspark.sql import SparkSession

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.4.1-bin-hadoop3"
os.environ["PATH"] += f":{os.environ['SPARK_HOME']}/bin"

spark = SparkSession.builder.appName("MySparkApp").getOrCreate()
```

---

## Summary

| Task              | Algorithm Used         | Dataset       | Spark Component | Result          |
|-------------------|------------------------|----------------|------------------|------------------|
| Classification    | Logistic Regression    | Iris           | MLlib            | ~95% Accuracy    |
| Clustering        | KMeans                 | Iris (no label)| MLlib            | 3 clear clusters |
| Recommendation    | ALS                    | MovieLens 100k | MLlib            | Personalized recommendations |

---



