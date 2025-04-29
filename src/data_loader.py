import pandas as pd
import os
import zipfile
import chardet  # 인코딩 감지를 위해 추가

def get_all_csv_files(data_dir):
    files = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            files.append(os.path.join(data_dir, file))
        elif file.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(data_dir, file), 'r') as zip_ref:
                for name in zip_ref.namelist():
                    if name.endswith('.csv'):
                        extract_path = os.path.join(data_dir, name)
                        zip_ref.extract(name, data_dir)
                        files.append(extract_path)
    return files

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def load_all_data(data_dir):
    files = get_all_csv_files(data_dir)
    dfs = []
    for f in files:
        # 파일 인코딩 감지
        detected_encoding = detect_encoding(f)
        for enc in [detected_encoding, "utf-8", "cp949", "utf-8-sig"]:
            try:
                df = pd.read_csv(f, encoding=enc)
                # 헤더가 깨진 경우 경고
                if any('�' in str(col) for col in df.columns):
                    print(f"⚠️ 헤더 깨짐: {f} (인코딩: {enc}) - 파일 헤더를 직접 수정하세요.")
                break
            except Exception:
                continue
        else:
            print(f"❌ 파일 읽기 실패: {f}")
            continue
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
