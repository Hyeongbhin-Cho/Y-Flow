# Scripts 패키지 (scripts/)

모델과 데이터셋 준비용 실행 스크립트를 둔다.

## 1. 관련 README 링크

* [Y-Flow](../README.md)
* [Data](../data/README.md)

## 2. 파일 목록 및 요약

* `setup_wan.py`: Wan2.1 가중치 준비 및 로드 검증.
* `setup_resnet34.py`: torchvision ImageNet-1K ResNet-34 V1 가중치를 받아 프로젝트 체크포인트 경로에 저장.
* `setup_clevrer.py`: 공식 CLEVRER 영상·주석 다운로드, ZIP 압축 해제, split manifest 생성.
* `setup_clevrer_eval.py`: Mask R-CNN / visual-mask / PropNet 평가 아티팩트 다운로드와 원본 클립 객체·상호작용 검토.
* `export_clevrer_flow.py`: 이미 학습된 `runs/.../flowmatch/last.pt`를 `checkpoints/clevrer_flow/`로 복사.

## 3. 세부 명세

### setup_resnet34.py

```bash
python scripts/setup_resnet34.py
```

torchvision의 ResNet34_Weights.IMAGENET1K_V1 가중치를 다운로드해 `checkpoints/resnet34_imagenet1k_v1.pth`에 저장한다. 이후 CLEVRER 인식 모델은 이 로컬 state dict를 사용하며, 모델 빌드 시 네트워크 접근을 요구하지 않는다. `--output`으로 경로를 바꾸거나 `--force`로 다시 받을 수 있다.

가중치와 CLEVRER train split 준비 후에는 `python main.py recognition --mode train --run_name exp_02_sub_video_recognition --config configs/exp_02_sub_video_recognition.yaml`로 지도학습을 시작한다. 개발 split에서 선택한 checkpoint는 `runs/exp_02_sub_video_recognition/recognition/best.pt` 및 `checkpoints/clevrer_recognition/last.pt`에 저장된다.

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

### setup_realestate10k.py

Exp-02 에피폴라 실험용 개발 클립을 준비한다. 기본은 **train 10클립 × 8프레임, 0.25초 간격, 512×288**이다. 실제 영상의 정적 장면 여부는 다운로드 후 사람이 검토해야 하며 manifest의 `static_scene_review`는 `pending`으로 시작한다.

```bash
conda env create -f environment.yml -n yflow311
conda activate yflow311
python scripts/setup_realestate10k.py --dry-run
python scripts/setup_realestate10k.py

# 이미 확보한 공식 포즈와 원본 MP4로 네트워크 없이 준비
python scripts/setup_realestate10k.py --poses-dir /path/to/poses --videos-dir /path/to/videos
```

`--poses-dir` 아래에는 `train/*.txt` 또는 `test/*.txt`, `--videos-dir` 아래에는 `<clip_id>.mp4`를 둔다. MP4는 시간 원점이 유지된 원본 영상이어야 한다. 잘라낸 영상의 PTS를 0으로 재설정하면 카메라 timestamp와 맞지 않는다.
이 환경에는 YouTube EJS 지원이 포함된 `yt-dlp[default]`와 JavaScript 런타임 Deno가 들어 있다. 기존 환경에 수동 설치할 때는 Python 3.11 이상에서 `yt-dlp[default]`, `av`, `pillow`와 Deno 2.3 이상을 준비한다.

- 공식 포즈 아카이브를 받으며, PNG 추출 후 임시 원본 영상과 압축파일을 제거한다. 포즈 텍스트는 재시도에 사용한다.
- seed로 후보 순서를 고정하고 기본 최대 100개 후보에서 성공한 10개를 준비한다. 다른 split manifest와 같은 URL은 중복 선택하지 않는다.
- 720p 이하 MP4를 우선 선택하고, 없으면 720p 이하 포맷, 마지막으로 제한 없는 최고 포맷을 시도한다. 마지막 fallback은 720p보다 큰 원본을 받을 수 있지만 출력은 설정한 크기로 resize한다.
- 영상 삭제·접근 제한 등은 manifest의 `failures`에 남긴다. 형식 오류는 재시도하며, 이미 기록된 비공개 영상과 sampling window보다 짧은 클립은 다음 실행에서 건너뛴다. 10개 미달이면 비정상 종료하고 완료된 클립은 재사용한다. 전역 네트워크/포즈 다운로드 오류는 즉시 종료한다.
- 카메라 row를 최근접 선택하고 실제 디코딩 PTS와 대조한다. 기본 허용 오차는 20ms이며 중복 프레임과 누락은 거부한다.
- 가로·세로를 지정 크기로 직접 resize한다. crop/padding은 없으며 K와 spatial transform에 각각의 축척을 반영한다.
- 동일 root/split에서 전처리 설정을 바꾸는 것은 거부한다. 별도 root를 사용한다.
- 자동 선별은 기하 유효성·정적 장면을 보장하지 않는다. RealEstate10K 원본 포즈 오차와 시차는 후속 오라클에서 확인한다.

```text
datasets/realestate10k/
├── poses/{train,test}/*.txt
├── clips/train/<clip_id>/
│   ├── 0000.png ... 0007.png
│   ├── cameras.npz
│   └── source_camera.txt
└── manifests/train.json
```

2026-09-15: 취소된 CLEVRER 실험의 로컬 `datasets/clevrer/` 영상·주석·압축 캐시는 삭제했다. 위 CLEVRER setup 설명은 기존 코드의 사용 기록이다.
