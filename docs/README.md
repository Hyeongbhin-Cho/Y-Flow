# Docs 패키지 (docs/)

이 패키지는 Y-Flow 프로젝트의 연구 노트, 실험 프로토콜, 작성 규칙을 둔다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)

---

## 2. 파일 목록 및 요약
* `NOTICE.md`: 마크다운·코드 작성 규칙
* `FlowMatch.md`: 무제약 FM baseline 구현 목표
* `HardFlow.md`: HardFlow 구현 목표 체크리스트
* `YFlow.md`: YFlow 구현 목표 체크리스트
* `GuideFlow.md`: GuideFlow 구현 목표 체크리스트
* `SafeFlow.md`: 담당 외. 비움
* `UniConFlow.md`: PTZF 기반 certificate, slack QP, Swiss Roll 적용 범위
* `exp/exp_01_swiss_roll.md`: Swiss roll 비교 실험 프로토콜. 데이터는 `datasets/swiss_roll/default/`에 고정

---

## 3. 세부 명세

### NOTICE.md
*   **설명**: 수식 표기, README 양식, 담당 범위, 주석·설정 파일 규칙.

### FlowMatch.md
*   **설명**: 제약 없는 FM 학습·샘플 목표. Exp-01의 무제약 기준선.

### HardFlow.md
*   **설명**: terminal hard constraint sampling의 구현 목표. 코드 작성 방식은 적지 않음.

### YFlow.md
*   **설명**: $P$ warm start + terminal $h,C$ + 선형 보간의 구현 목표.

### GuideFlow.md
*   **설명**: CVF / CF / RFE 세 제약 주입 전략의 구현 목표. Exp-01 이식 방법과 모듈 ablation 실측 포함.

### SafeFlow.md
*   **설명**: 다른 담당. 내용을 채우지 않음.

### UniConFlow.md
*   **설명**: 논문의 일반 constrained generation을 Swiss Roll 부등식 제약에 적용한 구현 대응표.

### exp/exp_01_swiss_roll.md
*   **설명**: 2D Swiss roll 포인트 생성 실험. 비교는 무제약 FlowMatch + HardFlow, SafeFlow, UniConFlow, GuideFlow, YFlow. 점과 meta는 dump 쌍을 쓴다.
