# HardFlow

Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization  
(Zeyang Li, Kaveh Alim, Navid Azizan)

Training-free. 사전학습된 속도장 $v_t^\theta$는 고정하고, inference에서만 궤적을 조정한다.  
중간 경로가 아니라 **최종 샘플** $x_N$에 hard constraint $h(x_N)\le 0$을 건다.

---

## 구현 목표

- [ ] 사전학습 Flow Matching $v_t^\theta$를 학습 없이 불러와 샘플링한다.
- [ ] 매 스텝에서 예측된 최종 상태 $\hat x_N$에만 $h\le 0$과 비용 $C$를 적용한다. 중간 $x_t$는 feasible일 필요가 없다.
- [ ] 제어량(nominal sampler와의 차이)을 작게 유지해 분포가 크게 밀리지 않게 한다.
- [ ] 마지막 스텝에서 $x_N=\hat x_N^*$이면 $h(x_N)\le 0$이 성립한다.
- [ ] 초반 스텝은 제어를 끄고, $t\ge t_{\mathrm{on}}$에서만 제약을 켠다.
- [ ] Exp-01에서 같은 $x_0$ 시드·같은 평가 점으로 Safety / Tube·Core 위반 / MMD / 시간을 남긴다.
- [ ] Default, $n_{\mathrm{eval}}=4000$에서 Safety $\ge 0.99$를 목표로 한다.
