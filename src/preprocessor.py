import pandas as pd
import numpy as np

def preprocess(df):
    # 날짜 변환
    df['대여일자'] = pd.to_datetime(df['대여일자'], errors='coerce')
    # 결측값 처리 (범주형)
    for col in ['대여구분', '성별', '연령대']:
        if col in df.columns:
            df[col] = df[col].fillna('미상')
    # 결측값 및 이상치 처리 (수치형)
    for col in ['이용건수', '운동량', '탄소량', '이동거리', '이동시간']:
        if col in df.columns:
            # 문자열을 숫자로 변환, 변환 불가능한 값은 NaN 처리
            df[col] = pd.to_numeric(df[col], errors='coerce')
            med = df[col].median()
            df[col] = df[col].fillna(med)
            q1, q3 = df[col].quantile([0.01, 0.99])
            df[col] = np.clip(df[col], q1, q3)
    # 중복 제거
    df = df.drop_duplicates()
    # 필요 없는 컬럼도 삭제 가능 (예: 대여소명 등)
    return df
