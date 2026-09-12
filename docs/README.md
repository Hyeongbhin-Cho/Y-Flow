# Docs 패키지 (docs/)

이 패키지는 Y-Flow 프로젝트의 연구 노트, 실험 프로토콜, 작성 규칙을 둔다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)

---

## 2. 파일 목록 및 요약
* `NOTICE.md`: 마크다운·코드 작성 규칙
* `FlowMatch.md`: 무제약 FM baseline 구현 목표
* `GuideFlow.md`: GuideFlow 구현 목표 체크리스트
* `SafeFlow.md`: FMBF/CFMBF-QP, Euler/Dopri5, terminal filter의 Exp-01 구현
* `UniConFlow.md`: PTZF 기반 certificate, slack QP, Swiss Roll 적용 범위
* `HardFlow.md`: HardFlow 구현 목표 체크리스트
* `YFlow.md`: YFlow 구현 목표 체크리스트
* `exp/exp_01_swiss_roll.md`: Swiss roll 비교 실험 프로토콜. 데이터는 `datasets/swiss_roll/default/`에 고정
* `exp/exp_02_video.md`: Wan2.1 기반 Hard-Constraint 비디오 생성 실험 계획. 첫 프레임, 공간 궤적, 시간 가속도 통일 제약
* `exp/exp_02_sub_video_recognition.md`: Wan 생성 전 CLEVRER FlowMatch 인식 실험. 객체·경로·충돌 과제와 $h(S)\le 0$

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
*   **설명**: 논문의 training-free SafeFlow를 $H=0$ Swiss-roll 점 문제로 축소한 구현과 한계를 설명한다.

### UniConFlow.md
*   **설명**: 논문의 일반 constrained generation을 Swiss Roll 부등식 제약에 적용한 구현 대응표.

### exp/exp_01_swiss_roll.md
*   **설명**: 2D Swiss roll 포인트 생성 실험. 비교는 무제약 FlowMatch + HardFlow, SafeFlow, UniConFlow, GuideFlow, YFlow. 점과 meta는 dump 쌍을 쓴다.

### exp/exp_02_video.md
*   **설명**: Wan2.1 기반 제약 비디오 생성 실험. 통일된 $h(V) \le 0$ 규칙 하에서 training-free 제약 5개 기법과 Y-Flow의 Safety, FVD, 추론 속도를 비교한다.

### exp/exp_02_sub_video_recognition.md
*   **설명**: Wan 픽셀 생성 전에 CLEVRER 주석 공간에서 객체 인식·궤적·충돌을 FlowMatch로 복원하고, 과제별 hard constraint를 건다.
