# Model 패키지 (model/)

이 패키지는 Flow-Matching 속도장 $v_t^\theta(x,t)$과 CLEVRER 직접 인식 모델을 정의한다.

Exp-01 Swiss roll은 2D **좌표점**이므로 MLP를 쓴다. CNN/UNet은 이미지 격자용이며 여기 쓰지 않는다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)
*   [Data](../data/README.md)

---

## 2. 파일 목록 및 요약
* `base.py`: `VelocityNet` 인터페이스, `build_model`
* `time_embed.py`: scalar $t\in[0,1]$ sinusoidal embedding
* `cond_mlp.py`: GuideFlow CFG용 조건부 속도장 (intent / reward 임베딩과 null 토큰)
* `mlp.py`: 저차원 속도장 MLP
* `download.py`: Hugging Face Wan2.1 가중치 및 CLEVRER 평가 artifact 다운로드
* `clevrer_eval.py`: 공식 visual-mask / PropNet 예측 로더, COCO Mask R-CNN 추론, 속성·충돌 점수
* `clevrer_flow.py`: 비디오 조건 인식 속도장 $v_\theta(S_t,t,E(V))$ (`CLEVRERVelocityNet`)
* `clevrer_recognition.py`: ImageNet 사전학습 ResNet-34 공간 인코더, 객체 슬롯 decoder, 시간 Transformer 및 상태 예측 head
* `wan.py`: Wan2.1 Flow-Matching Video Transformer (`WanTransformer3DModel`) 및 3D VAE 래퍼 (`WanVelocityNet`, `build_wan_model`)

---

## 3. 세부 명세

### base.py

#### VelocityNet
*   **설명**: `forward(x, t) -> v`. `x, v`는 `[B, D]` (또는 고차원 `[B, C, T, H, W]`), `t`는 `[B]` 또는 `[B, 1]`.

#### build_model
*   **설명**: `cfg.model.name`으로 네트워크를 만든다. Exp-01은 `mlp`, Exp-02는 `wan2.1`, CLEVRER CFM 인식 모델은 `clevrer_flow`, 직접 ResNet 인식 모델은 `clevrer_resnet34`.

### time_embed.py

#### SinusoidalTimeEmbedding
*   **설명**: DDPM식 주파수. 입력 `t: [B]` → 출력 `[B, dim]` (`dim`은 짝수).

### mlp.py

#### VelocityMLP
*   **설명**: `[x, embed(t)]`를 MLP에 넣어 $v\in\mathbb{R}^d$를 낸다. 기본 `dim=2`, hidden `(64, 64, 64)`.

### cond_mlp.py

#### ConditionalVelocityMLP
*   **설명**: GuideFlow CFG를 위한 조건부 속도장 $v_\theta(x,t,c)$. 의도(intent: 앵커 또는 커맨드)와 보상(reward: EP 진행도)을 임베딩하고, 독립 조건 마스킹과 null 토큰을 지원한다.

### download.py

#### download_wan_model
*   **설명**: Hugging Face 허브(`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`)에서 모델 가중치를 지정된 로컬 디렉터리(`checkpoints/Wan2.1-T2V-1.3B`)에 1회 다운로드하여 영구 저장한다. 이미 다운로드된 경우 네트워크 요청 없이 로컬 가중치를 재사용한다.
*   서브컴포넌트 필터 지원: `core` (DiT+VAE, ~3GB), `transformer` (~2.6GB), `vae` (~400MB), `all` (전체 파이프라인, ~13GB).
*   CLEVRER 평가 artifact: `download_clevrer_artifact(name)` → `checkpoints/clevrer/`. `visual_masks`, `propnet_preds`, `mask_rcnn`, `propnet`.

### clevrer_eval.py

*   공식 parser JSON과 PropNet 예측을 원본 주석과 비교한다. 속성 집합 F1, 가시 객체 수 MAE, 충돌 이벤트(프레임 허용오차)를 계산한다.
*   `load_mask_rcnn` / `detect_mask_rcnn`: 다운로드한 COCO Mask R-CNN으로 임의 RGB 프레임을 검출한다. CLEVRER fine-tune 가중치가 아니다.

### clevrer_flow.py

#### CLEVRERVelocityNet
*   **설명**: packed 장면 상태 $S\in\mathbb{R}^{D}$와 클립 $V$를 받아 $v_t^\theta(S_t,t,E(V))$를 낸다. $E(V)$는 프레임 CNN + 시간 평균. 학습 가중치는 `runs/{run_name}/flowmatch/last.pt`와 `checkpoints/clevrer_flow/last.pt`에 같이 둔다.

### clevrer_recognition.py

*   **설명**: `[B,3,T,H,W]` RGB 클립에서 ImageNet 사전학습 ResNet-34의 stem–layer3 공간 특징을 뽑고, 객체 슬롯 및 temporal Transformer head로 속성·가시성·world 위치/속도·충돌 로짓을 직접 예측한다.
*   사전학습 가중치는 `python scripts/setup_resnet34.py`로 `checkpoints/resnet34_imagenet1k_v1.pth`에 준비한다. 모델 생성 자체는 로컬 파일을 읽으며 자동 다운로드하지 않는다.

### wan.py

#### WanVelocityNet
*   **설명**: `VelocityNet` 인터페이스를 상속하여 Wan2.1 3D Transformer 속도장 $v_t^\theta(z_t, t)$ 및 3D Causal VAE 인코딩/디코딩 메서드를 제공한다.

#### build_wan_model
*   **설명**: `cfg.model` 설정을 참조하여 로컬 체크포인트를 로드하고, 미존재 시 자동 다운로드 옵션(`auto_download: true`)을 통해 가중치를 내려받아 모델을 빌드한다.
