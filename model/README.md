# Model 패키지 (model/)

이 패키지는 Flow-Matching 속도장 $v_t^\theta(x,t)$ 구조를 정의한다.

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

---

## 3. 세부 명세

### base.py

#### VelocityNet
*   **설명**: `forward(x, t) -> v`. `x, v`는 `[B, D]`, `t`는 `[B]` 또는 `[B, 1]`.

#### build_model
*   **설명**: `cfg.model.name`으로 네트워크를 만든다. Exp-01은 `mlp`.

### time_embed.py

#### SinusoidalTimeEmbedding
*   **설명**: DDPM식 주파수. 입력 `t: [B]` → 출력 `[B, dim]` (`dim`은 짝수).

### mlp.py

#### VelocityMLP
*   **설명**: `[x, embed(t)]`를 MLP에 넣어 $v\in\mathbb{R}^d$를 낸다. 기본 `dim=2`, hidden `(64, 64, 64)`.

### cond_mlp.py

#### ConditionalVelocityMLP
*   **설명**: GuideFlow CFG를 위한 조건부 속도장 $v_\theta(x,t,c)$. 의도(intent: 앵커 또는 커맨드)와 보상(reward: EP 진행도)을 임베딩하고, 독립 조건 마스킹과 null 토큰을 지원한다.
