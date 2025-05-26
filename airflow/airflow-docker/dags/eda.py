import os
import pandas as pd
import shutil

def combine_images():
    fake_folder = '/opt/airflow/ai_image'
    real_folder = '/opt/airflow/real_image'
    # output_dir = '/opt/airflow/combined_images'

    # os.makedirs(output_dir, exist_ok=True)

    fake_images = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) if os.path.isfile(os.path.join(fake_folder, f))]
    real_images = [os.path.join(real_folder, f) for f in os.listdir(real_folder) if os.path.isfile(os.path.join(real_folder, f))]

    fake_labels = ['FAKE'] * len(fake_images)
    real_labels = ['REAL'] * len(real_images)

    df = pd.DataFrame({
        'filepath': fake_images + real_images,
        'label': fake_labels + real_labels
    })

    df_combined = df.drop(columns=['Unnamed: 0', 'file_name'], errors='ignore')

    df_combined['label'] = df_combined['label'].map({
        0: 'REAL',
        1: 'FAKE',
        'REAL': 'REAL',
        'FAKE': 'FAKE'
    })

    # # Safely move files across devices
    # for index, row in df_combined.iterrows():
    #     src = row['filepath']
    #     dst = os.path.join(output_dir, os.path.basename(src))
    #     try:
    #         shutil.move(src, dst)
    #     except Exception as e:
    #         print(f"Failed to move {src} to {dst}: {e}")
    #     df_combined.at[index, 'filepath'] = dst

    csv_path = '/opt/airflow/dags/combined_images.csv'
    df_combined.to_csv(csv_path, index=False)

    print(df_combined.head())
    print(df_combined.label.value_counts())
    print(f"Combined dataset saved with {len(df_combined)} images.")

if __name__ == "__main__":
    combine_images()