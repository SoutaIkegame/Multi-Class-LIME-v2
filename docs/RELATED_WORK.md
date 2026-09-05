# 多クラスLIME・対比的局所説明の関連研究

更新日: 2026-09-03

## 0. このページの目的

本ページは、多クラス分類に対するLIME、クラス間の対比説明、確率ベクトル全体の説明、Fisher判別分析を利用した局所説明について、研究上重要な先行研究を整理する。

特に、現在検討している次の方法が、既存研究とどこで重なり、どこに未解決部分があるかを明確にする。

> LIMEと同じ摂動点・局所カーネルを使い、ユーザーが指定した任意のクラス対 $(C_c,C_d)$ について、ブラックボックスのpairwise log-ratio $\log(p_c/p_d)$ を疎な局所線形モデルで直接近似する。

手法自体の詳しい定式化は [`docs/OVO_LIME_METHODS.md`](OVO_LIME_METHODS.md) を参照する。

## 1. 調査方針と注意

- 主要な根拠には、査読済みの会議論文・ジャーナル論文と、出版社または会議の公式ページを優先する。
- arXiv版しかない研究は、査読済み研究と区別して記載する。
- 「関連する構成要素が既出であること」と「現在の提案と同一のアルゴリズムが既出であること」は区別する。
- 文献検索で完全一致する研究が見つからなかったことは、先行研究が存在しないことの証明ではない。本ページの結論は、2026-09-03までに確認できた文献に基づく。

参考にしたLIMEサーベイの添付版は2025年3月のarXiv版だが、その最終版はxAI 2025の査読済み会議論文としてSpringerから出版されている。サーベイ自身は、新規性がある場合には未査読arXiv論文も調査対象へ含めたと明記している。そのため、サーベイの参考文献にarXivが多いこと自体は調査方針による。

- Knab et al., *Which LIME Should I Trust? Concepts, Challenges, and Solutions*, xAI 2025, CCIS 2577, pp. 28--52: <https://doi.org/10.1007/978-3-032-08324-1_2>

同サーベイはLIMEの課題を、locality、fidelity、interpretability、stability、efficiencyの5種類に整理している。ただし、多クラス説明を直接の中心課題として扱うLIME拡張は多くなく、主要例はLIMEtreeである。

## 2. 前提となる問題設定

多クラスのブラックボックス分類器が、摂動点 $z$ に対して、

$$
f(z)=(p_1(z),\ldots,p_n(z)),
\qquad
\sum_{k=1}^{n}p_k(z)=1
$$

を返すとする。

通常の分類LIMEは、説明するクラス $C_c$ の確率 $p_c(z)$ を目的変数として、クラスごとの局所サロゲートを学習する。この処理はブラックボックスをone-vs-restで再学習することではない。ただし、得られる説明が答える問いは、意味上、

$$
C_c \quad\text{vs}\quad \mathcal C\setminus C_c
$$

というone-vs-restになっている。

特定の2クラスを比較する量としては、次が考えられる。

$$
q_{c,d}(z)
=
\frac{p_c(z)}{p_c(z)+p_d(z)}
$$

$$
\ell_{c,d}(z)
=
\log\frac{q_{c,d}(z)}{1-q_{c,d}(z)}
=
\log\frac{p_c(z)}{p_d(z)}
$$

$q_{c,d}$ は $C_c,C_d$ の二択に再正規化した確率、$\ell_{c,d}$ はそのlog-oddsである。

## 3. 多クラスLIMEの問題を直接扱う研究

### 3.1 LIMEtree: 独立したクラス別説明の問題

Sokol and Flachは、通常のLIMEがクラスごとに別の局所サロゲートを学習するため、複数クラスの関係を理解しにくいと指摘している。

特に、クラス $C_c$ の説明は暗黙に $C_c$ 対「その他すべて」を表す。そのため、$p_c\leq0.5$ であっても、特定の1クラスが $C_c$ より高いのか、複数の他クラスの確率を合計した結果なのかを、その説明だけでは区別できない。

また、クラスごとに独立した疎なモデルを作ると、異なる特徴部分集合や異なる説明構造が選ばれ、複数の説明が競合・矛盾して見える可能性がある。線形回帰で確率を近似すると、サロゲート出力が $[0,1]$ を外れる問題もある。

LIMEtreeは、すべてのクラス確率を1本のmulti-output regression treeで同時に近似する。全クラスの説明が共通の木構造から生成されることにより、説明の構造的な一貫性を確保する。

- Sokol and Flach, *LIMEtree: Consistent and Faithful Surrogate Explanations of Multiple Classes*, Electronics 14(5), 929, 2025: <https://doi.org/10.3390/electronics14050929>

本研究との関係は次の通りである。

- 問題意識は直接重なる。
- LIMEtreeは全クラスを一つの木で説明する。
- 本研究案は、指定したクラス対を線形係数によって直接説明する。
- 「多クラスLIMEの問題を初めて指摘した」とは主張できない。
- multi-output treeとpairwise linear surrogateは、説明形式と説明対象の問いが異なる。

### 3.2 CLIMAX: LIME型の局所logistic explainer

CLIMAXは、LIMEと同様の局所摂動を使い、線形回帰ではなく局所的なロジスティック分類器を説明器として利用する。L-CLIMAXでは、ブラックボックス確率 $p_c$ のlogit、

$$
\log\frac{p_c}{1-p_c}
=
\log\frac{p_c}{\sum_{k\neq c}p_k}
$$

を重み付きRidge回帰で近似する。CE-CLIMAXではsoft labelに対するcross entropyを用いる。さらに、クラス不均衡を緩和する摂動生成と、influence functionによるサンプル選択を導入している。

- Nanavati and Prasad, *CLIMAX: An Exploration of Classifier-Based Contrastive Explanations*, IEEE CogMI 2023, pp. 49--58: <https://doi.org/10.1109/CogMI58952.2023.00017>

「contrastive」という名称だが、基本の目的変数は特定の $C_d$ ではなく、$C_c$ 対残り全クラスである。したがって、

$$
\log\frac{p_c}{1-p_c}
$$

と、本研究案の、

$$
\log\frac{p_c}{p_d}
$$

は区別できる。ただし、「LIME型の摂動上でlog-oddsを局所回帰する」という大枠は既出である。

## 4. 全クラス確率を同時に説明する研究

### 4.1 SLISEMAP: 局所多項ロジスティック回帰

SLISEMAPは、各データ点に対する局所説明と、データ全体の低次元埋め込みを同時に学習する。分類では、ブラックボックスが出す確率ベクトル全体を目的変数とし、局所white-box modelとして多項ロジスティック回帰を使う。

基準クラスを1つ定めると、多項ロジスティック回帰は $n-1$ 個の係数ベクトルで表現できる。予測分布とブラックボックス分布の差には二乗Hellinger距離を使い、局所モデルにはLasso正則化を利用する。

- Björklund, Mäkelä, and Puolamäki, *SLISEMAP: Supervised Dimensionality Reduction Through Local Explanations*, Machine Learning 112, 1--43, 2023: <https://doi.org/10.1007/s10994-022-06261-1>

本研究との関係は次の通りである。

- 多クラス確率を一つの整合した局所モデルで説明する考え方は既出である。
- $n-1$ 個の基準クラスlog-ratioから任意のペア差を復元する構造も、多項ロジスティック回帰に内在する。
- SLISEMAPは埋め込みと全データ点の局所モデルを共同学習する方法であり、LIMEのように1つの説明点周囲へその場で摂動を生成する方法ではない。
- ユーザーが指定したfoil classとの対比を主な出力とする手法でもない。

### 4.2 AIM: 複数クラスの同時説明と共有特徴選択

AIMは、複数の対象クラスに対するinstance-wiseな加法的説明を同時に学習する。説明係数を行列として持ち、行単位のgroup normを用いて、複数クラスで共通する特徴選択を促す。

- Vo et al., *An Additive Instance-Wise Approach to Multi-class Model Interpretation*, ICLR 2023: <https://openreview.net/forum?id=ho2VRvHjTg>

したがって、「全クラスに同じ特徴集合を使わせる」「group sparsityで共通特徴を選ぶ」という発想だけでは新規性にならない。一方、AIMはLIMEの摂動ごとに独立した局所サロゲートをフィットする方法ではなく、説明器を全体として学習する点が異なる。

## 5. 多クラス確率を相対量として扱う理論

### 5.1 Pairwise coupling

多クラス確率から、

$$
q_{c,d}
=
\frac{p_c}{p_c+p_d}
$$

を作る操作は、pairwise couplingの文献で以前から使われている。

- Wu, Lin, and Weng, *Probability Estimates for Multi-class Classification by Pairwise Coupling*, Journal of Machine Learning Research 5, 975--1005, 2004: <https://www.jmlr.org/papers/v5/wu04a.html>

したがって、$q_{c,d}$ 自体を新しい確率として提案することはできない。ただし、この量を局所説明の教師信号として使うことは別の研究課題である。

### 5.2 Shapley compositions: simplexとlog-ratio

多クラス確率ベクトルは、各成分を独立に扱える通常の実数ベクトルではなく、総和が1に制約されたsimplex上のデータである。Shapley compositionsは、Aitchison geometryを使って、この相対的・合成的な性質を保ったまま確率予測を説明する。

- Noé et al., *Explaining a Probabilistic Prediction on the Simplex with Shapley Compositions*, ECAI 2024, pp. 1124--1131: <https://doi.org/10.3233/FAIA240605>

この研究はSHAP系でありLIMEではないが、次の点で重要である。

- 多クラス確率では絶対値だけでなく比率が重要である。
- 独立したクラス別説明は、確率分布の制約や相対関係を失う可能性がある。
- log-ratioは多クラス確率を扱う自然な座標になる。
- 1つのペアだけを見る説明は特定の対比には適するが、確率分布全体の説明と同一ではない。

### 5.3 Distributional Values: スカラー確率説明の限界

Distributional Valuesは、分類器の出力が本来はカテゴリカル分布であるのに、従来のSHAPなどが1クラスのスカラー確率を説明しているという対象の不一致を指摘する。クラス反転を含む分布値そのものを追跡する説明を提案している。

- Franceschi et al., *Explaining Probabilistic Models with Distributional Values*, ICML 2024, PMLR 235, pp. 13840--13863: <https://proceedings.mlr.press/v235/franceschi24a.html>

これもLIMEではないが、「単一クラス確率の説明だけでは多クラス分類器の出力構造を十分に表さない」という問題設定を、より一般的な立場から支持する。

## 6. Fisher判別分析と分類サロゲートに関する研究

### 6.1 TERP: LDAを局所説明の近傍設計に利用

TERPは、LIMEに似た摂動型の局所線形説明手法である。LDAによる1次元射影を、説明対象クラスとそれ以外を分ける局所近傍・類似度の構築に利用し、その後に線形モデルを学習する。

- Mehdi and Tiwary, *Thermodynamics-Inspired Explanations of Artificial Intelligence*, Nature Communications 15, 7859, 2024: <https://doi.org/10.1038/s41467-024-51970-x>

したがって、「LIMEの処理にLDAを持ち込む」こと自体には先行例がある。ただし、TERPではLDAをサロゲートそのものとして置き換えるのではなく、主に局所性を定義するために使用する。Fisher方向そのものを説明係数とする現在のFisher-LIME案とは異なる。

### 6.2 透明モデルを用いた局所説明の正しさの評価

Rahnama et al.は、40個の表形式データセットで、LIME、SHAP、Local Permutation Importanceを、線形回帰、ロジスティック回帰、Naive Bayesのモデル固有な正解説明と比較した。分類では、説明対象関数としてlog-oddsをLIMEへ渡している。

- Rahnama et al., *Can Local Explanation Techniques Explain Linear Additive Models?*, Data Mining and Knowledge Discovery 38, 237--280, 2024: <https://doi.org/10.1007/s10618-023-00971-3>

この研究は、次の評価上の示唆を与える。

- LIMEへlog-oddsを目的変数として与えることは査読済み研究で実施済みである。
- ブラックボックスが透明な多項ロジスティック回帰なら、真のpairwise係数 $\theta_c-\theta_d$ を直接計算できる。
- 摂動によるDeletion/Preservationだけでは、out-of-distribution入力が生じ、説明の正しさを直接測ったことにならない場合がある。
- 合成データや透明モデルによるground-truth係数評価を併用すべきである。

**2026-09-03に実装・実行済み**（`src/run_groundtruth_experiment.py`）：黒箱を多項ロジスティック回帰に差し替え、各手法（OVR / Fisher hard / Fisher soft / Contrastive）の推定pairwise係数と真の $\theta_{c^*}-\theta_{c'}$ のSpearman順位相関を測定した。結果はContrastiveが全手法・全グリッドセルで統計的に有意に最も真の係数を復元できることを示した（詳細は`docs/RECENT_WORK.md`、生の大きさではなく順位相関を使う理由もRahnama et al.の方針に準拠）。Rahnama et al.自体はLIME・SHAP・Local Permutation Importanceを線形回帰・ロジスティック回帰・Naive Bayesで比較したものであり、本研究の「多クラスLIME拡張4手法をpairwise log-odds線形性のもとで比較する」という設定はRahnama et al.そのものではなく、その評価方法論を本研究の比較対象に適用したもの。

## 7. 安定性と説明評価に関する研究

### 7.1 S-LIME

S-LIMEは、摂動標本数と特徴選択の統計的不確実性を扱い、LIMEの説明安定性を改善する。多クラス固有の方法ではないが、摂動乱数を変えた説明の安定性を比較する際の主要なベースラインになる。

- Zhou, Hooker, and Wang, *S-LIME: Stabilized-LIME for Model Explanation*, KDD 2021: <https://doi.org/10.1145/3447548.3467274>

### 7.2 XAI評価の体系

Nauta et al.は、XAI手法の評価軸を体系化し、correctness、consistency、contrastivity、compactnessなどを整理している。本研究ではユーザー実験を行わないため、これらを機能的・構造的な代理指標へ落とす必要がある。

- Nauta et al., *From Anecdotal Evidence to Quantitative Evaluation Methods: A Systematic Review on Evaluating Explainable AI*, ACM Computing Surveys 55(13s), Article 295, 2023: <https://doi.org/10.1145/3583558>

## 8. 先行研究との重なり

| 現在の研究案の要素 | 主な先行研究 | 判断 |
|---|---|---|
| 通常LIMEのクラス別説明が暗黙のOVRになるという問題 | LIMEtree | 既出 |
| クラス別に独立した説明が異なる特徴構造を持つ問題 | LIMEtree | 既出 |
| $q_{c,d}=p_c/(p_c+p_d)$ | Pairwise coupling | 既出 |
| $\log(p_c/p_d)$というlog-ratio | 統計学、多項ロジスティック回帰、Shapley compositions | 既出 |
| LIME型摂動上でlog-oddsを局所回帰 | CLIMAX、Rahnama et al. | OVRまたは二値では既出 |
| 多クラス確率全体を局所多項ロジスティック回帰で近似 | SLISEMAP | 既出 |
| 全クラスで共通特徴をgroup sparsityにより選ぶ | AIM | 既出 |
| LDAをLIME型説明の処理へ導入 | TERP | 近縁例あり |
| 任意の指定ペアについて $\log(p_c/p_d)$ をLIMEの局所目的変数として直接学習 | 完全一致は未確認 | 差別化候補 |
| pairwise Ridge、pairwise Lasso、joint multinomial、Fisher hard/softを同一摂動上で比較 | 完全一致は未確認 | 実験的貢献候補 |
| 疎性とpairwise cycle consistencyのトレードオフを測定 | 完全一致は未確認 | 分析上の貢献候補 |

## 9. 推移性・加法的整合性に関する注意

真のpairwise log-ratioは、

$$
\ell_{c,d}+\ell_{d,e}=\ell_{c,e}
$$

を満たす。

また、すべてのペアに対して同じ摂動行列、局所重み、全特徴、Ridge正則化係数を使う場合、Ridge推定は目的変数に対して線形なので、推定係数にも、

$$
\hat\beta_{c,d}
+
\hat\beta_{d,e}
=
\hat\beta_{c,e}
$$

が成立する。したがって、「OVOを独立に学習すると必ず推移性が失われる」とは主張できない。

整合性が崩れ得るのは、ペアごとに異なるLasso特徴選択、摂動集合、カーネル、正則化、クリッピング、非線形サロゲートなどを使う場合である。研究上は、OVO一般の欠点ではなく、**ペア固有の疎なモデル選択とクラス間整合性のトレードオフ**として扱う。

## 10. 現時点で安全な研究上の位置づけ

今回確認した査読済み文献群では、次の組合せに完全一致する方法は確認できなかった。

1. 標準LIMEと同じ説明点周囲の摂動と局所カーネルを使う。
2. ユーザーが任意のfoil class $C_d$ を指定できる。
3. ブラックボックスの $\log(p_c/p_d)$ または $q_{c,d}$ を直接の教師信号にする。
4. 疎な線形係数を「$C_d$より$C_c$を相対的に有利にする特徴」として提示する。
5. ペア別忠実性、確率忠実性、安定性、疎性、クラス間整合性を同じ実験条件で評価する。

したがって、現時点では次のように位置づけるのが安全である。

> LIMEtreeが指摘した独立one-vs-rest説明の問題に対し、全クラスを一つの木で説明するのではなく、ユーザーが選んだクラス対のpairwise log-ratioを直接近似する局所線形サロゲートを構成する。その有効性と限界を、クラス別LIME、局所多項ロジスティックモデル、分類型サロゲート、Fisher型サロゲートとの比較によって検証する。

主張を避けるべき表現は次である。

- 「pairwise確率を初めて提案した」
- 「log-oddsをLIMEへ初めて導入した」
- 「多クラス確率を初めて同時に説明した」
- 「共通特徴選択を初めて導入した」
- 「OVOは必ず推移性を失う」

## 11. 比較実験への示唆

### 11.1 最低限の比較対象

1. 通常のクラス別LIME
2. 通常LIMEの係数差 $\hat\beta_c-\hat\beta_d$
3. $p_c-p_d$ の直接Ridge回帰
4. pairwise log-ratioの直接Ridge/Lasso回帰
5. OVO soft-label logistic surrogate
6. 局所多項ロジスティック回帰
7. Fisher hard/soft
8. 可能であればLIMEtree
9. 安定性比較ではS-LIME

$p_c-p_d$ の直接Ridge回帰は、同じ設計で学習した通常LIMEの係数差と理論上一致する。そのため、実装検証用のnegative controlとして使える。

### 11.2 推奨する評価指標

- pairwise log-ratioの局所重み付きMSE
- $q_{c,d}$ に戻した後のBrier scoreまたはsoft-label log-loss
- $p_c>p_d$ の符号一致率とbalanced accuracy
- $q_{c,d}\approx0.5$ の境界付近に限定した一致率
- 摂動seed間の係数cosine類似度
- top-$K$特徴集合のJaccard類似度
- cycle residual
- 表示特徴数と計算時間
- 多項ロジスティック回帰をブラックボックスにした真の係数差 $\theta_c-\theta_d$ との誤差

全クラス確率を同時に出す手法については、二乗Hellinger距離またはKL divergenceによる分布全体の忠実性も測る。一方、Fisher方向の生の大きさには確率回帰係数と同じ単位がないため、係数の大きさや分散を未正規化のまま手法間比較しない。

## 12. 査読済み文献と未査読文献の区別

### 査読済みで主要な根拠に使う文献

- Ribeiro, Singh, and Guestrin, *Why Should I Trust You? Explaining the Predictions of Any Classifier*, KDD 2016: <https://doi.org/10.1145/2939672.2939778>
- Wu, Lin, and Weng, *Probability Estimates for Multi-class Classification by Pairwise Coupling*, JMLR 2004: <https://www.jmlr.org/papers/v5/wu04a.html>
- Zhou, Hooker, and Wang, *S-LIME: Stabilized-LIME for Model Explanation*, KDD 2021: <https://doi.org/10.1145/3447548.3467274>
- Björklund, Mäkelä, and Puolamäki, *SLISEMAP*, Machine Learning 2023: <https://doi.org/10.1007/s10994-022-06261-1>
- Vo et al., *An Additive Instance-Wise Approach to Multi-class Model Interpretation*, ICLR 2023: <https://openreview.net/forum?id=ho2VRvHjTg>
- Nauta et al., *From Anecdotal Evidence to Quantitative Evaluation Methods*, ACM Computing Surveys 2023: <https://doi.org/10.1145/3583558>
- Nanavati and Prasad, *CLIMAX*, IEEE CogMI 2023: <https://doi.org/10.1109/CogMI58952.2023.00017>
- Rahnama et al., *Can Local Explanation Techniques Explain Linear Additive Models?*, Data Mining and Knowledge Discovery 2024: <https://doi.org/10.1007/s10618-023-00971-3>
- Franceschi et al., *Explaining Probabilistic Models with Distributional Values*, ICML 2024: <https://proceedings.mlr.press/v235/franceschi24a.html>
- Noé et al., *Explaining a Probabilistic Prediction on the Simplex with Shapley Compositions*, ECAI 2024: <https://doi.org/10.3233/FAIA240605>
- Mehdi and Tiwary, *Thermodynamics-Inspired Explanations of Artificial Intelligence*, Nature Communications 2024: <https://doi.org/10.1038/s41467-024-51970-x>
- Sokol and Flach, *LIMEtree*, Electronics 2025: <https://doi.org/10.3390/electronics14050929>
- Knab et al., *Which LIME Should I Trust?*, xAI 2025: <https://doi.org/10.1007/978-3-032-08324-1_2>

### 参考にはするが、主要な新規性根拠にしない文献

LIPExは、確率分布全体を局所softmaxモデルで近似するため非常に近い発想を持つ。しかし、確認できた公開版はpreprintであり、ICLR 2024ではWithdrawn Submissionになっている。修論では関連案として触れてもよいが、「査読済みの確立した先行手法」として扱わない。

- Zhu et al., *LIPEx: Locally Interpretable Probabilistic Explanations to Look Beyond the Top Classes*: <https://openreview.net/forum?id=R7946uagL2>

