# PedalCast

서울시 공공자전거 대여소별 수요 예측 및 자전거 재배치 추천 시스템

- data/ : 여러 csv 또는 zip 파일 형태의 원본 데이터
- src/  : 데이터 처리, 모델링, 예측 등 파이썬 코드
- main.py : 전체 프로세스 실행
- 요구 패키지는 requirements.txt 참고

실행:
pip install -r requirements.txt
python main.py

결과: redistribution_suggestion.csv 로 추천 결과 출력