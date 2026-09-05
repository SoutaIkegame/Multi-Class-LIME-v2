# 実験ログ（詳細版）

更新日: 2026-09-05

<!--
この文書は今までに実施した全実験の詳細な記録。PROJECT_SUMMARY.md/RECENT_WORK.md
が「現在の状態・直近の引き継ぎ」を簡潔にまとめるのに対し、こちらは各実験の
目的・方法・生データに基づく数値・結論を時系列かつ網羅的に記録する。
数値は全て results/*.csv から再集計したもの（推定ではない）。
-->

## 0.0 訂正記録（2026-09-05、レビューにより発覚）

以下5点は、外部レビューで指摘され検証の上で訂正した。誤った記述は残さず、該当箇所を本文で直接修正している（この節は訂正の存在を明示するための索引）。

1. **フェーズ9〜11のfidelityは訓練に使った摂動`Z`上で測っていた（in-sample）**。未知の近傍への汎化を示す証拠にはならない。→ `run_combined_bc_experiment.py`にheld-out（独立に引き直した`Z_test`）評価を追加し、フェーズ11.5として再検証結果を追記した。
2. **「Fisherである限りギャップは消えない」は証明されていない**。フェーズ6で確認したのは特定の近傍・重み付け・正則化のもとでの結果であり、Fisherの一般的な限界を示したものではない。該当箇所を「この設定では」という限定付きの表現に修正した。
3. **提案A（フェーズ8）の敗因説明が実装と食い違っていた**。フェーズ7のFisher-select/Ridge-selectも実際には全クラスの情報を集約しており、「着目ペアだけの共有 vs 全クラス共有」という対比は誤り。正しい違いは「事後的なヒューリスティック集約（フェーズ7）vs 結合最適化（フェーズ8）」であり、本文を修正した。
4. **「ソフトラベル・ロジスティック損失は有界」は数学的に誤り**。有界なのは損失ではなく、スコアに対する勾配$\sigma(s)-q\in[-1,1]$。該当箇所を修正した。
5. **「有意差なし」を「固有の価値なし」、観測結果を確定した原因として書いていた箇所を、観測と仮説を区別する表現に修正した**（フェーズ6の「サンプル飢餓」、フェーズ7の「Fisherに固有の価値はない」、フェーズ9の「副作用のない改善」など）。

## 0.1 訂正記録（2026-09-06、2回目のレビューにより発覚）

1. **「直接回帰だから優れる」という説明は不正確**。Ridge回帰は目的変数に対して線形演算子（$\hat\beta=(Z^\top WZ+\lambda I)^{-1}Z^\top Wy$）なので、同一設計・重み・正則化なら$\mathrm{Ridge}(p_c)-\mathrm{Ridge}(p_d)=\mathrm{Ridge}(p_c-p_d)$が恒等的に成立する（`src/check_identities.py`で機械精度まで確認済み）。つまり「OVRの係数差」と「$p_c-p_d$の直接回帰」は数式上同一の推定量であり、「引き算 vs 直接回帰」はContrastiveを特徴づける違いではない。**本当の変更点は目的変数の変換（確率差 → 対数比）である**。`src/surrogates.py`の`fit_contrastive`docstring、本ログのフェーズ4を修正。
2. **循環整合性は「何も強制しない」のではなく、密なRidge版では数式上厳密に成立する**（線形性の帰結、`check_identities.py`で確認済み）。崩れるのは`fit_contrastive_lasso`でペアごとに特徴選択・実効正則化が変わるときのみ。
3. **`fit_contrastive`と`fit_ovo_logistic`のepsilon平滑化が不整合だった**（分母のみに加算 vs 分子分母に加算）。$q_\varepsilon=(p_{c_1}+\varepsilon)/(p_{c_1}+p_{c_2}+2\varepsilon)$に統一し、$\mathrm{logit}(q_\varepsilon)=\log((p_{c_1}+\varepsilon)/(p_{c_2}+\varepsilon))$が厳密に一致するようにした（`fit_ovo_logistic`修正、`check_identities.py`に検証追加）。
4. **中心的な研究の問いを精緻化**：「特定の競合クラスとの違いを説明するとき、通常のクラス別LIME（OVR）より、ペアを直接近似した方が、少ない特徴で忠実に説明できるか」を主軸に据え直す（詳細は`docs/OVO_LIME_METHODS.md`参照）。

## 0. 共通の実験設計

### 0.1 合成データとブラックボックス

- 合成データ: `sklearn.datasets.make_classification`。`n_informative = max(3, n_classes)`、`n_redundant = n_features - n_informative`（相関の強い冗長特徴量を意図的に含める）、`class_sep=1.2`、`n_samples=2000`、train/test分割はtest_size=0.3・random_state=0固定。
- ブラックボックスは実験ごとに2種類使い分ける。
  - **RandomForestClassifier**（`n_estimators=200, random_state=0`）: 非線形決定境界。ほとんどの実験（consistency, stability, fidelity, feature overlap, 極端領域, 提案A/B/Cの評価）で使用。
  - **LogisticRegression（multinomial）**（`max_iter=2000, C=1.0`）: 対数オッズについて厳密に線形。真の係数$\theta_c$が既知になるため、グラウンドトゥルース検証・Fisher診断・提案A/Cの一部評価で使用。

### 0.2 グリッドと摂動

- グリッド: `n_features ∈ {8, 14, 20}` × `n_classes ∈ {3, 4, 5}`（9セル）。実験によりKも `{0.3, 0.6} × n_features` の2水準を追加。
- 各セルで「最も際どい」8インスタンス（黒箱の予測確率で1位と2位の差が最小のもの、`pick_contested_instances`）を説明対象にする。
- 摂動: LIMEのデフォルト方式（`z = x + noise * feature_std`、指数カーネルで近接度重み付け、`src/perturbation.py`）を全手法・全実験で共有。

### 0.3 統計的枠組み（2026-09-03導入）

- 各グリッドセルにつき**独立したデータセット抽選を20回（`N_DATASET_SEEDS=20`）**繰り返す。
- 8インスタンスの平均をまずseedごとに取り（擬似反復の回避）、20個のseedレベル平均を独立サンプルとして扱う。
- 手法間比較は**対応ありWilcoxon符号順位検定**（同一seed・同一摂動を共有するため）。p値は**Holm-Bonferroni補正**（同一指標内の全セル・全手法ペアを1つの検定族として補正）。
- 実装: `src/stats_utils.py`（`bootstrap_ci`, `paired_wilcoxon`, `holm_bonferroni`, `compare_methods`）。

以下、各フェーズの実験を時系列で記録する。表の「有意」列は上記Holm補正後に有意だったグリッドセル数（分母は検定族のセル×手法ペア数）。

---

## フェーズ1: OVR vs Fisher LIME（consistency・stability）

**目的**: 従来のOVR形式のLIME（クラスごとに独立回帰）に対し、Fisher判別分析（LDA）を局所サロゲートとして使う代替案の妥当性を検証する（当初の研究提案そのもの）。

**手法**: `fit_onevsrest`（重み付きRidge回帰、クラスごと独立）、`fit_fisher`（プールされたクラス内散布行列$S_W$を使うLDA、ハードラベル版）。

**スクリプト**: `src/run_experiment.py` → `results/experiment_results.csv` / `experiment_stats.csv`

| 指標 | OVR | Fisher(hard) | 差（OVR−Fisher） | 有意 |
|---|---|---|---|---|
| feature overlap（Jaccard、K別×2水準） | 0.446 | 0.509 | −0.063 | 8/18 |
| stability（正規化分散） | 0.051 | 0.126 | −0.074 | 9/9 |

**結論**: Fisherはfeature overlapで数値上優位（後述フェーズ4で「クラス数増加につれ有意性が消失、逆転はしない」と精緻化）。stabilityは全セルで有意にFisherが劣る（Fisherの分散はOVRの約2.4倍）。

**理論修正の記録**: 当初「推移律が崩れる」という問題提起をしていたが、任意の実数の大小比較は常に推移的なため数学的に成立しないと判明（`transitivity_violation_rate`は理論的に常に0、コードのみ残置）。consistencyの正しい定義はLIMEtree（Sokol & Flach 2025, Sec.3, p.5）の「モデル間で共通構造・特徴部分集合を共有しない」ことに基づくと整理し直した。

---

## フェーズ2: Fidelity実験（絶対確率）

**目的**: Fisherを標準の多クラス確率分類器として評価した場合の忠実性を測る。

**手法**: OVR、Fisher(hard)、Fisher(soft)（`fit_fisher_soft`、ハードラベル化せず$\pi_i \cdot f_c(z_i)$を重みに使う）。損失は重み付き二乗Hellinger距離（SLISEMAP Eq.11と同じ）。

**スクリプト**: `src/run_fidelity_experiment.py` → `results/fidelity_results.csv` / `fidelity_stats.csv`

| 比較 | 平均損失差 | 有意 |
|---|---|---|
| OVR (0.017) vs Fisher hard (0.065) | −0.049 | 9/9 |
| OVR (0.017) vs Fisher soft (0.036) | −0.020 | 9/9 |
| Fisher hard (0.065) vs Fisher soft (0.036) | +0.029 | 9/9 |

**結論**: OVRがFisher(hard)より約3.9倍、Fisher(soft)より約2.2倍低損失（優れる）。Fisher soft はhardの約1.8倍改善するが、OVRには届かない。

---

## フェーズ3: 極端確率領域実験

**目的**: 競合する2クラスの一方の確率がほぼ0の局所領域（閾値0.15未満、局所近傍の平均19%を占める）でfidelityを測る。OVR・Fisher(hard)・Contrastive LIME（後述）の3手法。fidelityは「黒箱が決めた予測クラス$c^*$と競合クラス$c'$のペア比較の符号一致率」。

**スクリプト**: `src/run_extreme_regime_experiment.py` → `results/extreme_regime_results.csv` / `extreme_regime_stats.csv`

| 領域 | 比較 | 差 | 有意 |
|---|---|---|---|
| 極端領域 | Contrastive (0.944) vs OVR (0.937) | +0.007 | 6/9 |
| 極端領域 | Fisher (0.907) vs OVR (0.937) | −0.030 | 9/9 |
| 穏やかな領域 | Contrastive (0.782) vs OVR (0.786) | −0.004 | 3/9 |
| 穏やかな領域 | Fisher (0.771) vs OVR (0.786) | −0.015 | 3/9 |

**結論**: Contrastiveは極端領域で弱いが本物の優位性（6/9セルで有意）。Fisherは極端領域で一貫して有意に劣化——feature overlap・stabilityで見えた「ハードラベルのサンプル飢餓」が3つ目の文脈で再確認された。

---

## フェーズ4: Contrastive LIME追加（3手法同時比較）

**目的**: $\log(p_A/p_B)$を目的変数にするContrastive LIME（`fit_contrastive`、ユーザー提案）を追加し、OVR・Fisher(hard)・Contrastiveの3手法をfidelity・stability・feature overlapで同時比較する。

**注記（2026-09-06訂正）**：「OVRは2本フィットして引き算、Contrastiveは最初から直接回帰する」という当初の説明は不正確。Ridge回帰は目的変数に対して線形演算子なので、同一設計・重み・正則化なら「OVRの係数差」と「$p_c-p_d$の直接回帰」は数式上同一の推定量（0.1節参照）。Contrastiveが実際に変えているのは目的変数そのもの（確率差$p_c-p_d$ → 対数比$\log(p_c/p_d)$）であり、「直接 vs 引き算」という当てはめ方の違いではない。

**スクリプト**: `src/run_contrastive_experiment.py` → `results/contrastive_results.csv` / `contrastive_stats.csv`

| 指標 | 比較 | 値 | 差 | 有意 |
|---|---|---|---|---|
| fidelity（ペア符号） | OVR (0.821) vs Contrastive (0.819) | +0.001 | 1/9 |
| fidelity（ペア符号） | Fisher (0.800) vs Contrastive (0.819) | −0.020 | 4/9 |
| fidelity（ペア符号） | OVR (0.821) vs Fisher (0.800) | +0.021 | 5/9 |
| stability（正規化） | OVR (0.051) vs Contrastive (0.051) | +0.0001 | 4/9 |
| stability（正規化） | Fisher (0.126) vs Contrastive (0.051) | +0.075 | 9/9 |
| feature overlap | Fisher (0.509) vs Contrastive (0.466) | +0.043 | 3/18 |
| feature overlap | OVR (0.446) vs Contrastive (0.466) | −0.020 | 6/18 |

**結論**: ContrastiveはfidelityでOVRとほぼ完全な同点（差0.001〜0.002、ほぼ非有意）。stabilityもOVRとほぼ同格、Fisherより明確に優れる。feature overlapは中間的（クラス間 vs ペア間という比較単位の不一致は未解消）。

---

## フェーズ5: グラウンドトゥルース復元実験（Rahnama et al. 2024型）

**目的**: 黒箱をLogisticRegression（multinomial）に差し替え、真の係数$\theta_{c^*}-\theta_{c'}$が既知の状況で、各手法の推定係数とのSpearman順位相関を測る。局所再現性（fidelity）ではなく「正しい特徴に重みを置けているか」を検証する、質的に異なる評価軸。

**スクリプト**: `src/run_groundtruth_experiment.py` → `results/groundtruth_results.csv` / `groundtruth_stats.csv`

| 手法 | 平均 Spearman ρ |
|---|---|
| **Contrastive** | **0.9993** |
| Fisher (soft) | 0.9832 |
| OVR | 0.9758 |
| Fisher (hard) | 0.9528 |

全ペア比較が9/9セルで有意（Contrastive vs 他3手法、Fisher soft vs hard、OVR vs Fisher hard）。唯一 OVR vs Fisher soft のみFisher softがわずかに優る場合がある（差−0.007、8/9で有意）。

**結論**: Contrastiveが全手法に対し全セルで統計的に有意に最も真の係数を復元する——このプロジェクトで最も明確な勝ちどころ。理論的にも整合的（黒箱が対数オッズについて線形なら、対数オッズを直接回帰するContrastiveはほぼ完璧に一致するはず）。

---

## フェーズ6: Fisher方向の失敗要因分解（診断）

**目的**: Fisherの方向$v=S_W^{-1}(\mu_c-\mu_d)$を部品ごとに分解し、なぜ回帰系（OVR/Contrastive）に負けるのかを機構レベルで特定する。

**変種**（黒箱=LogisticRegression、真の係数との比較）:
- `centroid`: 重心差$\mu_c-\mu_d$のみ（$S_W$なし）
- `pooled`: 現行Fisher（全クラス共通$S_W$）
- `pair`: 2クラスだけで$S_W$を作る（OVO-Fisher）
- `diag`: $S_W$の対角成分のみ（回転なし）
- `cov_logodds`: 重心の代わりに対数オッズとの共分散を使う連続応答版

**スクリプト**: `src/diagnose_fisher_direction.py` → `results/diagnose_fisher_direction_results.csv` / `_stats.csv`

| 問い | 比較 | 差 | 有意 | 解釈 |
|---|---|---|---|---|
| $S_W^{-1}$は必要か | pooled_hard (0.953) vs centroid_hard (0.885) | +0.068 | 9/9 | 必要（尺度補正が効く） |
| $S_W^{-1}$は必要か | pooled_soft (0.983) vs centroid_soft (0.907) | +0.076 | 9/9 | 同上 |
| 全クラスプーリングの代償 | pair_hard (0.953) vs pooled_hard (0.953) | +0.0004 | 0/9 | **代償ほぼゼロ** |
| 全クラスプーリングの代償 | pair_soft (0.986) vs pooled_soft (0.983) | +0.003 | 4/9 | ほぼゼロ |
| 回転（非対角）は必要か | diag_soft (0.933) vs pooled_soft (0.983) | −0.050 | 9/9 | 必要（対角だけでは不十分） |
| ハード vs ソフト | pooled_soft (0.983) vs pooled_hard (0.953) | +0.030 | 9/9 | **サンプル飢餓が最大の損失源** |
| 重心 vs 連続応答 | cov_logodds (0.985) vs pooled_soft (0.983) | +0.001 | 4/9 | 改善はごく僅か |
| Contrastiveとの残差 | Contrastive (0.999) vs cov_logodds (0.985) | +0.015 | 9/9 | **本質的な差、消えない** |
| Contrastiveとの残差 | Contrastive (0.999) vs pooled_soft (0.983) | +0.016 | 9/9 | 同上 |

**結論**: (1) $S_W^{-1}$のスケール補正は必要で効いている。(2) 全クラス共通$S_W$というFisherの「売り」（クラス横断の共有構造）は精度をほぼ犠牲にしていない。(3) hard→softで有意に改善する（観測事実）。ソフトラベル化はハードラベル化で起きるクラス欠落（サンプル飢餓）を解消する設計であり、この観測と整合するが、原因を完全に切り分けたとまでは言えない（他の要因が寄与している可能性を排除していない）。(4) 連続応答にしても改善は僅少——重心という要約自体が情報を捨てているわけではない。(5) **残るギャップ（$\rho$にして0.985→0.999）は、今回の設定（この近傍サンプリング・このラベル重み付け・この正則化）では埋まらなかった、という限定的な結果である**。黒箱が対数オッズについて線形なとき、その量を直接回帰するContrastiveが有利なのは数式上自然（$\log(p_c/p_d)=(\theta_c-\theta_d)^\top z+\text{const}$を直接ターゲットにしているため）。これは「Fisherがこの設定でlog-ratio係数の復元に劣った」という結果であり、「Fisherは原理的にこの目的に使えない」という一般的な限界を証明したものではない——比較しているのは説明対象量が異なる2つの推定量（クラス内散布ベースの重心距離 vs log-oddsの回帰係数）であり、これ以上の一般化は現時点のデータからはできない。

---

## フェーズ7: 共有支持集合（二段構成）実験

**目的**: フェーズ6の(2)から、Fisherに残る役割は「係数の精度」ではなく「クラス横断の共有構造の供給」だと分かったため、二段構成（Stage1で共有特徴集合を選び、Stage2でContrastive回帰を再フィット）を評価する。正直な対照として、Stage1をFisher方向ではなく素のOVR Ridge係数の集約に置き換えた版（`ridge_select`）も比較する。

**手法**: `pair_lasso`（per-pair Contrastive Lasso、独立選択）、`fisher_select`（soft Fisherの方向を集約→Contrastive再フィット）、`ridge_select`（OVR Ridge係数を集約→Contrastive再フィット）。

**スクリプト**: `src/run_shared_support_experiment.py` → `results/shared_support_{logistic,rf}_*.csv`

### 黒箱=LogisticRegression（真top-K再現率・Spearman）

| 比較 | 差 | 有意 |
|---|---|---|
| fisher_select (0.749) vs ridge_select (0.741) の再現率 | +0.008 | 0/18 |
| ridge_select (0.741) vs pair_lasso (0.793) の再現率 | −0.053 | 5/18 |
| fisher_select (0.798) vs ridge_select (0.792) のSpearman | +0.006 | 0/18 |
| ridge_select (0.792) vs pair_lasso (0.817) のSpearman | −0.025 | 5/18 |

### 黒箱=RandomForest（ペア符号fidelity・支持集合安定性）

| 比較 | 差 | 有意 |
|---|---|---|
| fisher_select (0.784) vs ridge_select (0.784) のfidelity | +0.0003 | 0/18 |
| ridge_select (0.784) vs pair_lasso (0.761) のfidelity | +0.023 | 14/18 |
| fisher_select (0.789) vs ridge_select (0.789) の支持集合安定性 | +0.0004 | 0/18 |
| ridge_select (0.789) vs pair_lasso (0.717) の支持集合安定性 | +0.072 | 10/18 |

（ペア横断overlapは共有手法は構成上1.0、per-pair Lassoは平均0.466）

**結論**: (1) **Fisher-selectとRidge-selectの間に、検証した条件（両黒箱・全18セル×指標）では統計的な差は検出されなかった**（有意差0/18が一貫）。これは「両者が同等である」ことの証明ではなく、「今回のサンプルサイズ・条件下では優位性を確認できなかった」という結果であり、共有集合の選び方としてFisherに固有の利点がある可能性を完全には排除しない。(2) 共有集合そのもの（供給源不問）はper-pair Lassoに対しRFでfidelity・安定性が明確に優れる一方、ロジスティック黒箱の真top-K再現率で−5pt程度のコストを払う。「一貫性を買ってペア固有の精度を犠牲にする」というトレードオフが定量化できた（ただしこのfidelityも訓練に使った摂動上で測っており、フェーズ11.5の限界が同様に当てはまる可能性がある——未検証）。

---

## フェーズ8: 提案A「Multi-task Contrastive LIME」（不採用）

**目的**: フェーズ7の「二段構成」をさらに一歩進め、全クラスの対数確率を1本の多出力回帰で**同時学習**し、$\ell_{2,1}$（group lasso）で共有支持集合を学習段階から統合する。$\beta_{cd}=\gamma_c-\gamma_d$とパラメータ化すれば循環整合性（$\beta_{ab}+\beta_{bc}=\beta_{ac}$）は構成上ゼロになる。

**手法**: `fit_joint_contrastive`（`MultiTaskLasso`、$y_c=\log p_c - \text{mean}_k \log p_k$、二分探索で$K$個以上の非ゼロ行を持つ最疎解を探索）。`joint_lasso`＝生の係数、`joint_refit`＝選ばれた支持集合上でContrastive再フィット。

**スクリプト**: `src/run_shared_support_experiment.py`（`joint_lasso`/`joint_refit`列追加）

### 黒箱=LogisticRegression

| 比較 | 差 | 有意 |
|---|---|---|
| joint_refit (0.587) vs pair_lasso (0.793) の再現率 | −0.206 | 18/18 |
| joint_refit (0.587) vs ridge_select (0.741) の再現率 | −0.153 | 15/18 |
| joint_refit (0.705) vs pair_lasso (0.817) のSpearman | −0.113 | 17/18 |
| joint_refit (0.705) vs ridge_select (0.792) のSpearman | −0.088 | 14/18 |

### 黒箱=RandomForest

| 比較 | 差 | 有意 |
|---|---|---|
| joint_lasso (0.723) vs pair_lasso (0.761) のfidelity | −0.039 | 17/18 |
| joint_refit (0.769) vs ridge_select (0.784) のfidelity | −0.015 | 13/18 |
| joint_refit (0.178) vs ridge_select (0.096) の方向分散 | +0.082 | 14/18（不安定化） |
| joint_refit (0.709) vs ridge_select (0.789) の支持集合安定性 | −0.080 | 13/18 |

**結論**: **全指標・両黒箱で明確に劣る**（真の係数復元で−15〜21pt、fidelityで−1.5〜3.9pt、安定性も悪化）。**この案は不採用**。

**敗因の説明（訂正版）**: 当初「全クラスへの共有制約が着目ペアの精度を犠牲にする、Fisherと同じ共有しすぎ問題」と説明していたが、これは誤り。フェーズ7の`shared_support_fisher_soft`/`shared_support_ridge`も実際には**全クラスの方向・係数を集約して**共有特徴集合を決めており、「着目ペアだけに絞った共有」ではない——フェーズ7とフェーズ8は共に全クラスの情報を使っている点で同じ。両者の実際の違いは共有の**やり方**にある：
- フェーズ7は各クラスを**独立に**最適化した後（`fit_fisher_soft`のonevsrest方向、または`fit_onevsrest`のRidge係数）、事後的にヒューリスティックで特徴集合だけを集約し、係数の値自体はStage 2で着目ペアだけを使って独立に再フィットする（他クラスに一切引っ張られない）。
- フェーズ8（提案A）は全クラスの係数を**1つの結合最適化**（joint MultiTaskLasso）で同時に決めるため、特徴選択も係数の値も、フィッティングの最中に他クラスとのトレードオフに直接晒される。

「独立最適化の事後集約」と「結合最適化」のどちらがこの性能差の真因かは、本実験だけでは完全には切り分けられていない（有力な仮説として記録するに留める）。

---

## フェーズ9: 提案B「対比認識カーネル」

**目的**: 着目ペア$(c^*,c')$の摂動重みを、2クラスが拮抗している領域（$q=p_{c^*}/(p_{c^*}+p_{c'})\approx0.5$）に集中させる。$\pi'_i=\pi_i\cdot(4q_i(1-q_i)+\text{floor})$（floor=0.05）。

**注記**: 当初「極端領域を狙う」という設計意図で提案したが、$4q(1-q)$は$q=0.5$で最大・$q\to0,1$で最小になるため、実際には**極端領域を軽視し、拮抗領域を重視する**カーネルになっている（設計意図の記述ミスだったが、結果は解釈可能で一貫している）。

**スクリプト**: `src/run_pair_kernel_experiment.py` → `results/pair_kernel_results.csv` / `_stats.csv`（黒箱=RF）

| 指標 | standard | pairkernel | 差 | 有意 |
|---|---|---|---|---|
| 全体fidelity | 0.815 | 0.820 | +0.005 | **9/9** |
| 穏やかな領域fidelity | 0.783 | 0.790 | +0.007 | **9/9** |
| 極端領域fidelity | 0.952 | 0.953 | +0.0003 | 1/9 |
| 方向の安定性（分散） | 0.052 | 0.049 | −0.003 | 6/9 |

**結論**: 拮抗領域を重視するだけで、全体fidelityと安定性が一貫して改善（全体・穏やかな領域は全セルで有意）。極端領域は無風（悪化もしない）。**ここでの「コストなし」は、検証した3指標（全体/極端/穏やかなfidelity、方向の安定性）の範囲で悪化が見られなかったという意味であり、他の未検証の副作用がないことの証明ではない**。

**重要な限界（フェーズ11.5で対応）**: このfidelity・extreme・moderateは、いずれも**サロゲートの学習に使った摂動`Z`そのもの**の上で符号一致率を測っている（in-sample）。これは学習内の適合度としては有効な指標だが、未知の近傍でも同じ改善が再現される証拠にはならない。独立に引き直した摂動での再検証はフェーズ11.5を参照。

---

## フェーズ10: 提案C「OVO local logistic」

**目的**: Contrastiveの目的変数$\log(p_{c_1}/p_{c_2})$は同じまま、損失関数をRidge（対数変換後の二乗誤差、$q\to0,1$付近で発散）からソフトラベル交差エントロピー$L(s,q)=\log(1+e^s)-qs$に変える。**注記（訂正）**：この損失自体は$s\to\infty$で発散し有界ではない。有界で$q=0,1$付近でも飽和するのはスコア$s$に対する**勾配**$\sigma(s)-q\in[-1,1]$の方であり、「損失が有界」という当初の記述は誤りだった。

**手法**: `fit_ovo_logistic`（各行を$y=1$（重み$\pi_i q_i$）と$y=0$（重み$\pi_i(1-q_i)$）に複製し、`LogisticRegression`に`sample_weight`で渡す——重み付きソフトラベル交差エントロピーの標準的な実装）。

**スクリプト**: `src/run_logistic_target_experiment.py` → `results/logistic_target_{logistic,rf}_*.csv`

### 黒箱=LogisticRegression（ほぼ天井効果）

| 指標 | logistic | ridge | 差 | 有意 |
|---|---|---|---|---|
| Spearman | 0.998 | 0.999 | −0.0017 | 8/9 |
| 真top-K再現率 | 0.987 | 0.996 | −0.0093 | 1/9 |

### 黒箱=RandomForest

| 指標 | logistic | ridge | 差 | 有意 |
|---|---|---|---|---|
| 全体fidelity | 0.819 | 0.815 | +0.0045 | 8/9 |
| 穏やかな領域fidelity | 0.789 | 0.783 | +0.0057 | 8/9 |
| 極端領域fidelity | 0.953 | 0.952 | +0.0013 | 1/9 |
| 方向の安定性（分散） | 0.048 | 0.052 | −0.0040 | **9/9** |

**結論**: フェーズ9（B）とほぼ同じパターン（全体・穏やかな領域fidelity改善、極端領域無風）だが、**安定性の改善はBより一貫している（9/9 vs 6/9）**。真の係数復元はリッジよりわずかに劣るが、両方0.99台の天井効果でほぼ無視できる。単体で見ると4案の中で最もバランスが良い。**RF黒箱側のfidelity/extreme/moderateはフェーズ9と同じくin-sample測定であり、同じ限界を持つ（フェーズ11.5参照）**。

---

## フェーズ11: B+C 組み合わせ

**目的**: BとCは直交する改良（カーネル vs 損失関数）で個別に似た効果を示したため、組み合わせて積み上がるか検証する。

**手法**: `standard`（無印）、`kernel`（B単体）、`logistic`（C単体）、`combined`（B+C）。

**スクリプト**: `src/run_combined_bc_experiment.py` → `results/combined_bc_*.csv`（黒箱=RF）

| 指標 | standard | kernel | logistic | combined | combined vs standard | combined vs logistic |
|---|---|---|---|---|---|---|
| 全体fidelity | 0.815 | 0.820 | 0.819 | **0.821** | +0.007（9/9有意） | +0.002（6/9有意） |
| 穏やかな領域fidelity | 0.783 | 0.790 | 0.789 | **0.792** | +0.009（9/9有意） | +0.003（6/9有意） |
| 極端領域fidelity | 0.952 | 0.952 | 0.953 | 0.952 | 変化なし | 変化なし |
| 方向の安定性（分散） | 0.052 | 0.049 | **0.048** | 0.050 | わずかに改善（2/9のみ有意） | **悪化**（0.0019、6/9有意） |

**結論（訂正前・in-sample測定）**: fidelityは素直に積み上がる（combined > kernel単体・logistic単体、いずれもstandardからの改善は最大）。**しかし安定性は打ち消し合う**——logistic単体が持っていた「境界付近でも勾配が飽和して暴れない」という利点（仮説、フェーズ10参照）を、カーネルで拮抗領域をさらに重視すると逆に効きすぎてブレが増えるように見える。**fidelityを取るか安定性を取るかで最適な組み合わせが変わる**、という交互作用が観測された。

**重要な限界**: 上記のfidelity/extreme/moderateは全て、学習に使った摂動`Z`そのもの上で測定したin-sample値である（`_sign_acc`の呼び出しが学習と評価で同じ`Z, proba, w`を使っていた）。学習内の適合度としては有効だが、未知の近傍でも同じ改善が再現される証拠にはならない——特にカーネル系（B・combined）は学習時に境界付近のサンプルを重視するため、その同じサンプルで評価すれば見かけ上fidelityが上がりやすい構造になっている。independent held-outでの再検証結果は次節（フェーズ11.5）。

---

## フェーズ11.5: 忠実性のheld-out再検証（訂正、2026-09-05）

**目的**: フェーズ9〜11のfidelity/extreme/moderateは学習に使った摂動`Z`上のin-sample測定だった（外部レビューで指摘）。同じインスタンス・同じカーネルで独立に引き直した摂動`Z_test`で測り直し、in-sample版の結論がどこまで生き残るかを確認する。

**手法**: `run_combined_bc_experiment.py`を改修し、`Z`で学習した後、新規に`sample_perturbations`を呼んで得た`Z_test`（学習に一切使っていない）でfidelity/extreme/moderateを測る`*_test`列を追加。standard/kernel(B)/logistic(C)/combined(B+C)の4手法を同時に再評価。方向の安定性（`direction_variance`）はもともと独立resamplingで測っていたため対象外・再検証不要。

**スクリプト**: `src/run_combined_bc_experiment.py` → `results/combined_bc_*.csv`（`_train`＝旧来のin-sample値、`_test`＝held-out値）

### in-sample（`_train`）vs held-out（`_test`）の平均値比較

| 指標 | standard | kernel(B) | logistic(C) | combined(B+C) |
|---|---|---|---|---|
| 全体fidelity（train） | 0.811 | 0.815 | 0.815 | 0.817 |
| 全体fidelity（**test**） | 0.800 | 0.803 | 0.803 | 0.804 |
| 穏やかな領域fidelity（train） | 0.780 | 0.786 | 0.785 | 0.788 |
| 穏やかな領域fidelity（**test**） | 0.769 | 0.773 | 0.773 | 0.775 |

いずれの手法もtrain→testで0.01〜0.012ポイント下落しており、in-sample測定には確かに楽観的なバイアスがあった（想定通り）。

### held-out（`_test`）での統計的な優劣（20シード、Holm補正）

| 比較 | 全体fidelity | 穏やかな領域fidelity | 極端領域fidelity | 方向の安定性 |
|---|---|---|---|---|
| kernel(B) vs standard | +0.0035（4/9有意） | +0.0045（4/9有意） | +0.0005（0/9） | −0.0032（3/9有意） |
| logistic(C) vs standard | +0.0033（6/9有意） | +0.0041（5/9有意） | +0.0013（1/9） | −0.0043（8/9有意） |
| combined vs standard | +0.0045（5/9有意） | +0.0058（5/9有意） | +0.0009（0/9） | −0.0025（2/9有意） |
| **combined vs kernel(B)単体** | **+0.0010（0/9）** | **+0.0013（0/9）** | +0.0004（0/9） | +0.0007（4/9有意、悪化） |
| **combined vs logistic(C)単体** | **+0.0012（0/9）** | **+0.0017（1/9）** | −0.0004（0/9） | +0.0018（6/9有意、悪化） |

**結論（held-out版、これが正）**:

1. **B・Cそれぞれ単体のfidelity改善は、規模は小さくなるが held-out でも概ね生き残る**（standard比、全体・穏やかな領域ともに4〜6/9セルで有意）。in-sampleほど鮮やかではないが、「弱いが本物」の改善として扱ってよい。
2. **「B+Cを組み合わせるとfidelityがさらに積み上がる」という、フェーズ11のin-sample結論は held-out では再現されなかった**。combinedとkernel単体・logistic単体の差は、全体fidelityで0/9、穏やかな領域fidelityでも0〜1/9しか有意にならない——in-sample版で見えていた上乗せ（6/9・6/9で有意）は、大部分ないし全部が学習サンプルへの適合度の見かけ上の差だった可能性が高い。
3. **極端領域fidelityは train・test 問わず一貫して無風**（0〜1/9）。B・Cとも極端領域への効果は確認できていない。
4. **方向の安定性（resamplingベースのため元々in-sampleの問題を受けない）は変わらず**：logisticが単体で最も一貫して改善（8/9）、combinedはlogistic単体より悪化する（6/9で有意）——フェーズ11の結論通り。

**修正した採用方針**: 「B+Cで積み上がる」という主張は撤回する。**単体で見て一番安定して効果があるのはC（OVO logistic）**で、fidelity・安定性ともにheld-outで確認できる改善を持つ唯一の案。Bを追加するかどうかは、held-outでは追加の恩恵がほぼ確認できない（かつ安定性をわずかに損なう可能性がある）ため、**Cのみを主軸として提示し、B併用は積極的には推奨しない**、という結論に修正する。

---

## フェーズ12: OVO vs OVR、同等の表示特徴数での忠実性比較（2026-09-06）

**目的**: 精緻化した中心的な問い「特定の競合クラスとの違いを説明するとき、OVRより少ない特徴で忠実に説明できるか」を直接検証する。OVR-union（$c_1,c_2$それぞれ独立にtop-K Lasso選択し、係数差の台＝2集合の和集合を複雑さとする）、Contrastive（`fit_contrastive_lasso`、ペアで共同選択した厳密にK個）、Logistic/提案C（`fit_ovo_logistic_lasso`、新規実装、L1正則化ロジスティック回帰でペア共同選択した厳密にK個）の3方式を、**学習に使っていない独立摂動`Z_test`**でのペア符号一致率で比較する（フェーズ11.5の教訓を最初から反映）。

**手法**: `fit_ovo_logistic_lasso`（新規、`src/surrogates.py`）：`fit_ovo_logistic`と同じソフトラベル交差エントロピーを、L1正則化ロジスティック回帰（`penalty="l1", solver="liblinear"`）でK個ちょうどに疎化する。二分探索は`fit_contrastive_lasso`と同じ「K個以上の非ゼロ係数を持つ最疎解」方式（`C`＝逆正則化強度なので探索方向は`_sparsest_lasso_with_at_least_k`と逆）。

**スクリプト**: `src/run_ovo_vs_ovr_experiment.py` → `results/ovo_vs_ovr_{results,stats}.csv`（黒箱=RF、9セル×K2水準×20シード）

### 結果（held-out、全セル・全K平均）

| 比較 | 差 | 有意セル数（Holm後） |
|---|---|---|
| Contrastive vs OVR-union | −0.0007 | **0/18**（ほぼ完全な同点） |
| Logistic(C) vs OVR-union | −0.0055 | 2/18（両方Cが劣る方向） |
| Logistic(C) vs Contrastive | −0.0048 | 3/18（全てCが劣る方向） |

**重要な非対称性**：OVR-unionの実際の複雑さ（選択特徴数の和集合）は指定Kの**約1.3〜1.4倍**（例：K=2で平均2.8〜3.1、K=10で平均12.7〜13.6）。2クラスそれぞれ独立にK個選ぶ以上、和集合がKちょうどになることは通常なく、OVRは複雑さの面で有利な条件を与えられている。

**結論（正直に、これまでの想定より弱い）**:

1. **「同じ表示特徴数ならOVOの方が忠実」という中心的な問いは、疎（Lasso/L1選択）な設定では支持されなかった**。Contrastiveは複雑さで不利な比較（OVRは1.3〜1.4倍多い特徴を使用）でもOVR-unionとほぼ完全な同点——「明確に勝つ」とは言えないが、より少ない実効特徴数で同等の忠実性に達しているとも解釈できる、両義的な結果。
2. **Logistic（提案C）は疎な設定でOVR-union・Contrastiveの両方にわずかに劣る**（3/18・2/18で有意、全て同じ方向）。これは**フェーズ11.5（密なRidge/logistic、Lasso選択なし）でlogisticがContrastiveより有意に優れていた held-out結果と表面的に矛盾する**。整合的な解釈は「logisticの優位性は密な全特徴フィットに限定され、L1による疎化を経由すると消える、あるいは逆転する」というもの——ソフトラベル交差エントロピーのL1正則化パスが、Ridgeの$\ell_2$正則化パスと同じようには振る舞わない可能性がある（未検証の仮説）。
3. フェーズ11.5とフェーズ12は「密 vs 疎」という異なる条件を比較しているため、両者は矛盾する2つの真実ではなく、**「どちらの設定を採用するかで結論が変わる」**という、指標依存性の追加の一例として記録する。

**今回のスコープ外**（2026-09-06レビューの仮説2・3、`docs/OVO_LIME_METHODS.md`参照）：
- 選択された特徴がクラス共通成分を避けペア固有成分を選べているか（precision/recall）は、共通/ペア固有/無関係特徴を明示的に作る合成データ生成器が必要で未実装。
- 競合クラスを変えると説明が適切に変わるかの検証も未実装。

### 追記：複雑さを揃えた対照実験（2026-09-06、同日中に追加）

**目的**：上記の結果はOVR-unionが指定Kの1.3〜1.4倍の特徴を使う「有利な条件」での比較だった。この有利さを取り除くと結論が変わるかを確認する。

**手法**：`ovr_union_half`（各クラス$\lceil K/2\rceil$個をLasso選択、和集合を取る）を追加。実際の複雑さは**Kの0.6〜0.8倍**（元の`ovr_union`のKの1.3〜1.4倍とは逆方向、正確に$K$に一致させたわけではなく「有利側」と「不利側」で挟む形）。

| 比較 | 差 | 有意セル数 |
|---|---|---|
| Contrastive vs ovr_union（有利側） | −0.0007 | 0/18（同点） |
| Contrastive vs ovr_union_half（不利側） | **+0.0453** | **18/18** |
| Logistic(C) vs ovr_union_half（不利側） | **+0.0405** | **14/18** |
| ovr_union vs ovr_union_half（参考：複雑さの効果自体） | +0.0460 | 18/18 |

**結論（訂正・強化）**：OVR側の複雑さを有利側（K比1.3〜1.4倍）から不利側（K比0.6〜0.8倍）に振ると、結論が「同点」から「OVOの明確な勝利」に反転する。真に$K$へ正確に一致させた場合の結果は未測定だが、有利側でようやく同点・不利側で明確な敗北という2点で挟まれていることから、**正確に複雑さを揃えればOVR-unionはContrastive・Cの両方に負ける可能性が高い**。フェーズ12本文の「中心的な問いは疎な設定では支持されなかった」という結論は、**OVR側に複雑さの有利さを与えたままの比較に基づく誤った結論だった**として訂正する。中心的な問い（同じ表示特徴数ならOVOの方が忠実か）は、**複雑さを公平に揃えれば支持される**、というのが現時点の結論。

残課題：$K$に正確に一致させる版（例：和集合サイズが$K$になるよう$K_1,K_2$を動的に調整する）は未実装。

---

## 13. 現時点の採用方針（2026-09-05時点、フェーズ12の結果を踏まえて要更新）

**注記（2026-09-06、同日中に訂正）**：フェーズ12の当初の結果（OVO vs OVR「同点」）は、OVR-unionが指定Kの1.3〜1.4倍の特徴を使う有利な条件での比較だったことが判明し、その有利さを取り除く対照実験（フェーズ12「複雑さを揃えた対照実験」）では**Contrastive・Cの両方がOVR-unionに明確に勝つ**結果になった。中心的な問いは複雑さを公平に揃えれば支持される、というのが最新の結論。ただし密/疎でのCの逆転（下記6.）自体は解消していない。

1. **提案手法 = Contrastive LIME**。真の係数復元（フェーズ5）・極端領域fidelity（フェーズ3）で全手法に対し統計的に有意に優位。
2. **改良版として提案C（OVO logistic）を単体の主軸**とする。fidelity・安定性の改善はheld-out再検証（フェーズ11.5）でも生き残っている（規模は小さくなる）。**提案Bの追加併用は積極的には推奨しない**——held-outではCへの上乗せがほぼ確認できず（0〜1/9で有意）、安定性をわずかに損なう可能性がある（フェーズ11.5で訂正）。
3. **一貫性が必要な場合の選択肢として「共有支持集合＋Contrastive再フィット」**（フェーズ7）を提示する。利得（fidelity・安定性・overlap=1）と代償（真top-K再現率−5pt程度）を定量化済み。供給源はRidge集約とFisher集約に統計的な差は検出されなかった（同等の証明ではない、フェーズ7参照）。**注**：このfidelityもin-sample測定であり、フェーズ11.5と同種の再検証が未実施。
4. **Fisher（LDA）は当初の提案から降ろし、分析章として位置づける**。係数推定器としてはこの設定で回帰系に本質的に勝てず（フェーズ6、一般的な限界の証明ではない）、共有構造の供給源としてもRidge集約との差は検出されていない（フェーズ7）。「なぜこの設定でLDA的発想が回帰に劣るか」を機構レベルで示した結果として記録する。
5. **提案A（Multi-task Contrastive LIME、全クラス同時のgroup lasso）は不採用**（フェーズ8）。真の係数復元・fidelity・安定性の全軸で明確に劣ることが確認された。敗因の具体的な機序（独立最適化の事後集約 vs 結合最適化）は仮説段階。
6. **フェーズ12（複雑さを揃えた版）**：疎な（表示特徴数を絞った）設定でも、複雑さを公平に揃えればContrastive・Cの両方がOVR-unionに明確に勝つ（18/18・14/18で有意）。**ただし、疎な設定でのC vs Contrastive自体は依然としてCがわずかに劣る**（3/18で有意、密な設定＝フェーズ11.5とは逆の序列）。「Cを主軸とする」（2番目）という方針は、この意味で依然として**密な全特徴フィットの場合に限定される**——疎な設定ではContrastiveの方が単体としては手堅い。修論の記述では「密/疎のどちらの設定を主張の中心に置くか」を明示する必要がある。

## 14. 未検証・今後の課題

- **フェーズ1〜8のfidelity（`run_experiment.py`, `run_contrastive_experiment.py`, `run_fidelity_experiment.py`, `run_extreme_regime_experiment.py`, `run_shared_support_experiment.py`のRF側）は、フェーズ9〜11と同じくin-sample測定のまま、held-out再検証は未実施**。フェーズ11.5でB/C/combinedについては再検証したところ、in-sample版の効果量・有意性が有意に縮小する（一部は消える）ことが分かったため、過去のフェーズの数値も同程度の縮小がある可能性がある。優先度高い持ち越し課題。
- 提案Aの敗因仮説（独立最適化の事後集約 vs 結合最適化のどちらが真因か）は未検証（フェーズ8参照）。
- ソフト版Fisherのstability（正規化）はフルグリッド未検証（3セルのみのアドホック検証）。
- `src/investigate_reversal.py`（n_classes=6〜7での「逆転」診断）は統計的枠組みへの移行未実施。
- 黒箱がラベルのみ返す状況（`predict_proba`なし）でのFisher(hard) vs OVR(0/1ラベル回帰)比較は未実施——Fisherに残る唯一の原理的な居場所候補。
- 実データセットでの再現性確認は未実施（合成データのみ）。
- インスタンス選択（マージン最小8点のみ）の妥当性検証は未実施。
- feature overlapの「クラス間 vs ペア間」比較単位の不一致は未解消。
- **フェーズ12で見つかった密/疎の逆転（特にlogisticの逆転）の原因は未調査**。ソフトラベル交差エントロピーのL1正則化パスの挙動を診断する必要がある。
- **OVR-unionの複雑さを正確に$K$へ一致させた比較は未実施**（K/2版は「有利側」と「不利側」で挟んだだけで、正確な一致ではない。和集合サイズが$K$になるよう動的に調整する版が望ましい）。
- フェーズ12で示唆された仮説2（共通特徴の回避）・仮説3（競合クラス変更への追随）は、合成データ生成器の新規実装が必要で未着手。

## 15. 実験スクリプト対応表

| スクリプト | 黒箱 | 対応フェーズ |
|---|---|---|
| `src/run_experiment.py` | RF | 1 |
| `src/run_fidelity_experiment.py` | RF | 2 |
| `src/run_extreme_regime_experiment.py` | RF | 3 |
| `src/run_contrastive_experiment.py` | RF | 4 |
| `src/run_groundtruth_experiment.py` | LogisticRegression | 5 |
| `src/diagnose_fisher_direction.py` | LogisticRegression | 6 |
| `src/run_shared_support_experiment.py` | RF + LogisticRegression | 7, 8 |
| `src/run_pair_kernel_experiment.py` | RF（fidelityはin-sample、参考値） | 9 |
| `src/run_logistic_target_experiment.py` | RF（fidelityはin-sample、参考値）+ LogisticRegression | 10 |
| `src/run_combined_bc_experiment.py` | RF（`_test`列がheld-out、正） | 11, 11.5 |
| `src/run_ovo_vs_ovr_experiment.py` | RF（held-out評価を最初から実装） | 12 |
| `src/stats_utils.py` | — | 統計基盤（フェーズ1以降共通） |
| `src/check_identities.py` | — | 恒等式チェック（OVR差分＝直接回帰、循環整合性、epsilon平滑化の整合性）。データに依存しない代数的な検証で、統計的検定の対象ではない |
| `src/investigate_reversal.py` | RF | （統計化未実施、フェーズ1派生の診断） |
