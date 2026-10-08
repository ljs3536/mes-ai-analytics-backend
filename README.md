# MES Analytics Backend

학습과 실시간 예측을 담당합니다. MLflow는 실험 기록과 모델 버전(레지스트리)만 맡고, 이 서비스가 그 모델을 로드해 MQTT 특징값을 판정합니다.

1. 수집기 `GET /api/features`로 최근 특징값을 가져와 Isolation Forest를 학습합니다.
2. 모델을 MLflow에 등록하고 `production` alias를 붙입니다.
3. `mes/machines/{code}/features`를 구독해 판정하고 `anomaly`를 발행합니다.
4. 이상이 연속 3회면 MES `POST /api/machines/by-code/{code}/hold`를 호출합니다.

MLflow UI(http://localhost:5000)만으로는 실시간 예측과 MES 정지가 되지 않습니다.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/uvicorn app.main:app --port 8003 --reload
```
