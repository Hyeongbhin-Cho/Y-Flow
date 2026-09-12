# Scripts 패키지 (scripts/)

모델과 데이터셋 준비용 실행 스크립트를 둔다.

## 1. 관련 README 링크

* [Y-Flow](../README.md)
* [Data](../data/README.md)

## 2. 파일 목록 및 요약

* `setup_wan.py`: Wan2.1 가중치 준비 및 로드 검증.
* `setup_clevrer.py`: 공식 CLEVRER 영상·주석 다운로드, ZIP 압축 해제, split manifest 생성.
* `setup_clevrer_eval.py`: Mask R-CNN / visual-mask / PropNet 평가 아티팩트 다운로드와 원본 클립 객체·상호작용 검토.
* `export_clevrer_flow.py`: 이미 학습된 `runs/.../flowmatch/last.pt`를 `checkpoints/clevrer_flow/`로 복사.

## 3. 세부 명세

### setup_clevrer.py

[공식 CLEVRER 배포 페이지](https://clevrer.csail.mit.edu/)의 MIT 서버를 사용한다. 기본은 validation 영상 5,000개와 정답 주석이며, 학습이나 모델 추론은 수행하지 않는다. 다운로드 자체는 Python 표준 라이브러리만 사용한다.

```bash
# 다운로드 주소와 저장 위치만 확인
python scripts/setup_clevrer.py --dry-run

# 기본 평가 split 준비
python scripts/setup_clevrer.py --splits validation

# 모든 split과 선택적인 질문/정답 파일
python scripts/setup_clevrer.py --splits train validation test --questions

# 저장 위치 변경 (상대 경로는 프로젝트 루트 기준)
python scripts/setup_clevrer.py --root datasets/clevrer --splits validation
```

* train/validation은 영상과 주석을 함께 받는다. 공개 정답 주석이 없는 test는 영상만 받는다.
* `--questions`를 주면 VQA JSON도 받는다. 객체 마스크·검출 모델은 다운로드하지 않는다.
* 공식 배포 단위인 split 전체 ZIP을 받으므로 디스크에는 ZIP과 압축 해제한 영상이 함께 저장된다.
* 기존 다운로드 파일은 재사용하고 중단된 `.part`는 서버가 HTTP Range를 지원하면 이어받는다. 지원하지 않으면 처음부터 다시 받는다.
* ZIP을 읽으며 CRC를 검사하고, 파일별 임시 경로에서 완료된 파일로 교체한다. 경로 이탈과 중복 scene 파일을 거부한다.
* split 파일 개수와 영상/주석 scene ID를 확인한 후 manifest를 기록한다. 손상된 기존 파일은 `--force`로 다시 다운로드·압축 해제한다.
* 기본 저장 위치는 실행 디렉터리와 무관하게 프로젝트의 `datasets/clevrer/`다.

```text
datasets/clevrer/
├── archives/                    # 원본 ZIP 및 압축 해제 완료 표시
├── videos/validation/video_10000.mp4
├── annotations/validation/annotation_10000.json
├── questions/validation.json    # --questions 사용 시
└── manifests/validation.json    # scene ID와 상대 경로 목록
```

MP4 디코딩에는 `environment.yml`에 명시한 PyAV(`av`)가 필요하다. 사용법은 [data/README.md](../data/README.md)의 CLEVRER 절을 참고한다.

### setup_clevrer_eval.py

생성 비디오의 물리 제약을 사후 검토하기 위한 공개 artifact를 `checkpoints/clevrer/`에 받는다. Wan 가중치와 같이 다운로드 로직은 `model/download.py`, 실행 스크립트는 `scripts/`에 둔다.

```bash
# 출처·용도만 확인
python scripts/setup_clevrer_eval.py --dry-run

# 공개 artifact 다운로드
python scripts/setup_clevrer_eval.py

# 원본 CLEVRER eval 100 clip에서 객체/충돌 인식 점수
python scripts/setup_clevrer_eval.py --skip-download --eval
```

| artifact | 실제 내용 | 새 영상(Wan)에 적용 |
| :--- | :--- | :---: |
| `visual_masks` | 공식 Mask R-CNN **parser 산출물** (`derender_proposals.zip`) | 불가. 원본 CLEVRER 전용 |
| `propnet_preds` | 공식 PropNet **사전 계산 궤적/충돌** | 불가. 원본 CLEVRER 전용 |
| `mask_rcnn` | torchvision COCO Mask R-CNN. CLEVRER fine-tune 가중치는 **미공개** | 가능. 도메인 불일치 가능 |
| `propnet` | DCL 공개 `.pth`. 공식 CLEVRER 학습 체크포인트는 **미공개** | 불가(raw RGB). tube proposal 필요 |

산출물: `checkpoints/clevrer/artifacts.json`, `eval_report.json`.

### export_clevrer_flow.py

학습이 `runs/{run_name}/flowmatch/last.pt`에만 있는 경우, 다른 실험이 읽을 고정 경로로 복사한다. 학습 루프는 `model.local_dir`이 있으면 매 저장마다 같은 복사를 한다.

```bash
# 기본 run_name과 yaml의 local_dir
python scripts/export_clevrer_flow.py --run_name exp_02_sub_video_recognition

# 경로를 직접 지정
python scripts/export_clevrer_flow.py --src runs/exp_02_sub_video_recognition/flowmatch/last.pt --dst checkpoints/clevrer_flow
```

산출물: `checkpoints/clevrer_flow/last.pt`, `config.yaml`, `READY.json`.
