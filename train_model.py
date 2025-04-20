import tensorflow as tf
import pandas as pd
import numpy as np

# UAE-specific synthetic data
data = {
    'age': np.random.randint(18, 70, 10000),
    'bmi': np.round(np.random.uniform(18, 45, 10000), 1),
    'glucose': np.random.randint(70, 300, 10000),
    'hba1c': np.round(np.random.uniform(4, 12, 10000), 1),
    'diabetes': np.random.choice([0, 1], 10000, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

model.fit(
    df[['age', 'bmi', 'glucose', 'hba1c']],
    df['diabetes'],
    epochs=20,
    validation_split=0.2,
    batch_size=32
)

model.save('diabetes_model.h5')
print("UAE Diabetes AI model trained successfully!")