import pandas as pd
import numpy as np
import json
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def load_split():
   with open('data_split.json', 'r') as f:
       split_dict = json.load(f)
   return np.array(split_dict['train_indices']), np.array(split_dict['test_indices'])

def construct_path(row):
   p_prefix = f"p{str(row['subject_id'])[:2]}"
   p_id = f"p{row['subject_id']}"
   s_id = f"s{row['study_id']}"
   return f"generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/files/{p_prefix}/{p_id}/{s_id}/{row['dicom_id']}.tfrecord"


def parse_tfrecord(raw_tf):
    raw_dataset = tf.data.TFRecordDataset([raw_tf])
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        embedding_feature = example.features.feature['embedding']
        embedding_values = embedding_feature.float_list.value

    return np.array(embedding_values)


def load_data_and_train():
   # Load original DataFrame
   df = pd.read_csv('cxr_ids_y.csv')
   df = df.drop(df.columns[0], axis=1)
   df['file_path'] = df.apply(construct_path, axis=1)

   # Load train/test split
   train_idx, test_idx = load_split()
   train_df = df.iloc[train_idx]
   test_df = df.iloc[test_idx]

   # Load and process TFRecords
   train_features = []
   train_labels = []
   test_features = []
   test_labels = []

   # Load training data
   for idx, row in train_df.iterrows():
       try:
           embedding = parse_tfrecord(row['file_path'])
           train_features.append(embedding)  # No need for .numpy() now
           train_labels.append(row['Y'])
       except Exception as e:
           print(f"Error processing training file {row['file_path']}: {str(e)}")

   # Load test data
   for idx, row in test_df.iterrows():
       try:
           embedding = parse_tfrecord(row['file_path'])
           test_features.append(embedding)
           
           test_labels.append(row['Y'])
       except Exception as e:
           print(f"Error processing test file {row['file_path']}: {str(e)}")

   # Convert to numpy arrays
   X_train = np.array(train_features)
   y_train = np.array(train_labels)
   X_test = np.array(test_features)
   y_test = np.array(test_labels)

   # Train XGBoost model
   model = xgb.XGBClassifier(
       max_depth=3,
       learning_rate=0.1,
       n_estimators=100,
       objective='binary:logistic',
       random_state=42
   )

   # Train
   model.fit(X_train, y_train)

   # Predictions
   y_pred = model.predict(X_test)
   y_pred_proba = model.predict_proba(X_test)[:, 1]

   # Calculate metrics
   accuracy = accuracy_score(y_test, y_pred)
   auc = roc_auc_score(y_test, y_pred_proba)
   f1 = f1_score(y_test, y_pred)

   print("Model Performance Metrics:")
   print(f"Accuracy: {accuracy:.4f}")
   print(f"AUC-ROC: {auc:.4f}")
   print(f"F1 Score: {f1:.4f}")

   return model, (accuracy, auc, f1)

if __name__ == "__main__":
   model, metrics = load_data_and_train()
