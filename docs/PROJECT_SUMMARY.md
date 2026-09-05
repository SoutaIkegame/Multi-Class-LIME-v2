# Project Summary

<!--
この文書はプロジェクト全体の現在像を示す長期的なまとめです。
一時的な作業履歴ではなく、別端末・別セッションでも必要になる恒久情報を記載してください。
-->

## プロジェクトの目的

多クラス分類問題におけるLIMEの拡張を検討する修論研究。従来のLIMEの多クラス対応（one-vs-rest：クラスごとに独立した線形サロゲートをフィットする）が抱える問題点（LIMEtreeの指摘：クラスごとに異なる特徴部分集合を使い、説明が矛盾しうる）を整理する。

**中心的な問い（2026-09-06に精緻化・確定）**：「黒箱の予測クラス$c^*$に対し、特定の競合クラス$c'$との違いを説明するとき、通常のクラス別LIME（OVR）より、そのクラス対$(c^*,c')$を直接近似する方式（OVO）の方が、少ない表示特徴数で忠実に説明できるか」。

当初はFisher判別分析（LDA）を代替の局所サロゲートとして提案・検証する計画だったが、実証の結果（フェーズ6・7、`docs/EXPERIMENT_LOG.md`参照）Fisherは分析章に位置づけを変更し、**提案手法はContrastive LIME（対数比$\log(p_{c^*}/p_{c'})$を直接近似するOVO方式）及びその改良版（ソフトラベル交差エントロピー版、"提案C"）**とする。経緯の詳細は`docs/EXPERIMENT_LOG.md`、手法の定式化一覧は`docs/OVO_LIME_METHODS.md`を参照。

## 現在の状態

- 理論設計は一通り完了。摂動生成・黒箱への問い合わせ・近接度重み付けはLIMEと共通のまま、「サロゲートをフィットする」ステップだけをFisher LDAに置き換える設計。
- one-vs-rest LIME vs Fisher LIMEのconsistency・stability比較実験を実装・完走済み（グリッド実験2回、診断実験1回）。
- **重要な理論修正あり**：当初の「推移律が崩れる」という問題提起は数学的に成立しないことが判明（任意の実数の大小比較は常に推移的なため）。
- **consistencyの正しい定義はLIMEtree（Sokol & Flach 2025）の一次資料に基づく**：「モデル同士が共通構造を共有しない・異なる特徴量部分集合を使う」ことが矛盾した説明の原因（p.5）。sum-to-oneや推移律そのものではない。
- **査読済み文献を中心に関連研究を再整理済み**（`docs/RELATED_WORK.md`）。pairwise条件付き確率、log-ratio、多クラス同時サロゲート、共有特徴選択、LDAの局所説明への導入にはそれぞれ先行研究がある。一方、「LIMEの局所摂動上でユーザー指定の任意クラス対の$\log(p_c/p_d)$を疎な線形モデルとして直接学習し、各代替手法と統一条件で比較する」という組合せに完全一致する査読済み手法は、2026-09-03時点の調査では未確認。
- **訂正記録（2026-09-05、外部レビューで発覚）**：以下5点を訂正済み（詳細と修正箇所は`docs/EXPERIMENT_LOG.md`冒頭「0.0 訂正記録」参照）。(1) 提案B/C/combinedのfidelityは学習に使った摂動`Z`上でのin-sample測定だった——held-out再検証を追加（`run_combined_bc_experiment.py`の`*_test`列、結果はEXPERIMENT_LOG.mdフェーズ11.5）。(2) 「Fisherである限りギャップは消えない」は未証明——「この設定でFisherが劣った」に限定する表現へ修正。(3) 提案A（不採用）の敗因説明「着目ペアだけの共有 vs 全クラス共有」は実装と矛盾——フェーズ7も全クラス共有のため、正しくは「独立最適化の事後集約 vs 結合最適化」の違い。(4) 「ロジスティック損失は有界」は数学的に誤り——有界なのは勾配。(5) 「有意差なし→固有の価値なし」「観測→確定原因」という言い切りを、観測と仮説を区別する表現に修正。以下の本文中、まだ修正しきれていない同種の表現がある場合はEXPERIMENT_LOG.mdの記述を優先する。
- **統計的厳密性の強化（2026-09-03）**：それまでの4本の実験ドライバは、各グリッドセルにつきデータセット・分類器の抽選が1回だけで、8インスタンスの平均を信頼区間・検定なしで報告していた（＝そのデータセットがたまたま非典型だった可能性を区別できなかった）。`src/stats_utils.py`を新設し、各グリッドセルを**独立したデータセット抽選20回（N_DATASET_SEEDS=20）**で反復し、seedごとにインスタンス平均を取ってから（擬似反復の回避）ブートストラップ信頼区間・対応ありWilcoxon符号順位検定・Holm-Bonferroni多重比較補正を適用する方式に、主要4実験（`run_experiment.py`, `run_contrastive_experiment.py`, `run_fidelity_experiment.py`, `run_extreme_regime_experiment.py`）を作り直した。以下の結論はこの統計的枠組みで再検証済み（詳細は`docs/RECENT_WORK.md`参照）。`src/investigate_reversal.py`（診断スクリプト）はこの反復方式にまだ移行しておらず、単一シードのままである点に注意。
- 実測で確認済みの結論（詳細は`docs/RECENT_WORK.md`参照）：
  - **stability（正規化ペア方向ベクトルの分散）**：**Fisher（ハード版）はone-vs-restより不安定**という結果が、20独立シード全てで一貫して再現された（全9グリッドセルでp≈0.000002、Holm補正後も有意）。ソフトラベル版のstabilityは今回のフルグリッド再検証の対象外のまま（依然としてアドホックな3セル検証のみ、フルグリッドでの再現は未実施）。
  - feature overlap・fidelity実験は尺度の問題を抱えていない（feature overlapは順位のみ使うため尺度不変、fidelityは両手法とも適切な単位の確率値に変換してから比較しているため）。
  - **【訂正】feature overlap（LIMEtreeの主張の直接的な操作化）**：以前は「クラス数が多いと逆転する（OVRが優位になる）」と記載していたが、統計的に厳密な再検証では、Fisherは検証した全18グリッドセル×K（n_classes=3〜5の範囲）で数値上は一貫してOVRより高い（優位）。ただし有意性はクラス数が増えるほど失われ、n_classes=5の多くのセルで非有意になる。**OVRが統計的に有意に上回るケースは1つも観測されなかった**——正確には「逆転」ではなく「優位性の消失（クラス数増加につれ差が検出できなくなる）」。より高いクラス数（6〜7）での真の逆転を主張する`investigate_reversal.py`の診断結果は、異なる設計（n_informative固定）かつ単一シードのままなので、この訂正と直接矛盾はしないが再検証が必要。
  - ソフトラベル版Fisher（`fit_fisher_soft`）はクラス欠落は解消するが、feature overlap自体は多くの条件でむしろ悪化する（クラス間の分離が弱まるため）。単純な優劣ではなくトレードオフとして扱うべき。
  - **fidelity（忠実性）は測り方で結論が変わる**。Fisherを標準の多クラス確率分類器として評価すると one-vs-rest に大きく劣る（Hellinger損失、全9セル×hard/soft版で統計的に有意にOVRが優位、p≈0.000002）。しかし提案アルゴリズムが実際に使う量（黒箱が決めた予測クラス$c^*$と競合クラス$c'$のペア比較の符号一致率）で測り直すと、OVRはFisher(hard)より9セル中4セルで有意に優位（n_classes・n_featuresが大きいセルに集中）——**「ほぼ互角」ではなく「小さいグリッドでは互角、大きいグリッドではOVRがわずかに有意に優位」**という、以前より精緻化された結論。**Fisherの忠実性の弱さは「絶対確率値としての解釈」でより顕著**、という切り分けは維持。
  - **現時点の全体像**：ハード版Fisherに明確な優位性はない（stability・fidelity(絶対値・ペア符号どちらも)のいずれも統計的に有意に劣る。feature overlapのみ数値上は優位を保つが有意性は不安定）。ソフト版はfidelity(絶対値)でハード版より統計的に有意に優れる（全9セル）が、stabilityはフルグリッド未検証のまま。「Fisherが勝つ」という単純な主張ではなく、**指標ごとに条件付きで一長一短がある**、という正直な立ち位置は変わらない。
  - **4手法目「Contrastive LIME」**：$\log(p_A/p_B)$を目的変数にする方式。**訂正（2026-09-06）**：「OVRを2本フィットして引く vs 直接回帰する」という当初の説明は不正確——Ridge回帰は目的変数に対する線形演算子なので、同一設計・重み・正則化なら両者は数式上同一の推定量になる（`src/check_identities.py`で機械精度まで確認済み）。実際に変えているのは**目的変数そのもの**（確率差$p_c-p_d$ → 対数比$\log(p_c/p_d)$）。fidelity（ペア符号一致率）はOVRとほぼ完全な同点（9セル中8セルで非有意、差は0.002〜0.004）——統計的に確認済み。stabilityはOVRとほぼ同格（3/9セルで有意にContrastiveがわずかに安定、それ以外は非有意）、Fisherより全9セルで有意に安定。feature overlapはOVRとほぼ非有意差（3/18で有意、小さい）、Fisherとは低クラス数・高次元セルで有意差あり（比較単位がクラス間 vs ペア間で異なる点は未解消）。
  - **極端確率領域での検証**：競合する2クラスの一方の確率が0に近い局所領域でfidelityを測ると、Contrastiveのone-vs-restに対する優位性は全9セル中6セルで統計的に有意（クラス数・次元数が大きいセルに集中、n_classes=3では非有意）。**新しい発見**：この優位性は無償ではなく、穏やかな領域ではContrastiveがOVRよりわずかに、しかし統計的に有意に劣る場合がある（9セル中3セルで有意、差は-0.002〜-0.008）——極端領域での優位性と引き換えに穏やかな領域で小さなコストを払っている可能性。一方、**Fisher(hard)はこの極端領域で全9セルにおいて統計的に有意に劣化**（p<0.003）しており、feature overlap・stabilityでも見られた「ハードラベルによるサンプル飢餓」が3つ目の独立した文脈・厳密な検定で再確認された。
  - **Fisher方向の失敗要因分解（2026-09-05、`diagnose_fisher_direction.py`）**：真の係数とのSpearman ρで、重心差のみ0.885/0.907（hard/soft）→ pooled $S_W^{-1}$ 0.953/0.983 → Contrastive 0.999。統計的に確定した内訳：(1) $S_W^{-1}$は必要（+0.05〜0.08、単位補正）。(2) **クラス横断のプーリングはほぼ無料**（ペア限定$S_W$との差≤0.005、多くのセルで非有意）——共有構造という売りは精度を犠牲にしていない。(3) **ハードラベルが最大の損失源**（soft化で+0.02〜0.04）。(4) 重心を連続応答（log-oddsとの共分散）に置き換えても+0.005程度。(5) **残りの差0.985→0.999は、この設定（この近傍サンプリング・このラベル重み付け・この正則化）では埋まらなかった**：黒箱が対数オッズについて線形なとき、その量を直接回帰するContrastiveが有利なのは数式上自然（$\log(p_c/p_d)=(\theta_c-\theta_d)^\top z+\text{const}$）。これは「Fisherがこの設定でlog-ratio係数の復元に劣った」という限定的な結果であり、Fisherの一般的な限界を証明したものではない（説明対象量が異なる2推定量の比較のため、これ以上の一般化は現時点のデータからはできない）。**暫定的な帰結**：Fisherの残された役割候補は「係数」ではなく「共有構造（特徴集合）」。$S_W$をRidgeの罰則行列にする案（`OVO_LIME_METHODS.md`のFisher-metric系）は、この尺度偏りを回帰に持ち込むだけなので不採用。
  - **共有支持集合（二段構成）の評価（2026-09-05、`run_shared_support_experiment.py`、20シード、Holm補正）**：per-pair Contrastive Lasso（独立選択）vs Fisher-select（提案）vs Ridge-select（対照）を固定Kで比較。**(a) Fisher-select と Ridge-select の間に、検証した条件では統計的な差が検出されなかった**（両黒箱・全18セル×指標でHolm有意ゼロ、差は再現率+0.008、fidelity+0.0003程度）。これは同等性の証明ではなく、今回の条件下では優位性を確認できなかったという結果。**(b) 共有集合そのものは、RF黒箱では per-pair Lasso に対して明確に優れる**：ペア符号fidelity +0.023（18セル中15〜16で有意、小さいKほど差が大きく最大+0.07）、特徴集合の再サンプリング安定性（Jaccard）+0.072（12〜13/18で有意）、ペア横断overlapは構成上1.0（Lassoは0.47）。方向の安定性は一貫した差なし（大きいKで2/18セルのみLassoが有意に安定）。**(c) 共有の代償はロジスティック黒箱の真top-K再現率で−0.044〜−0.053（6/18で有意）、Spearmanで−0.02〜−0.025（3〜5/18で有意）**：1つの集合を全ペアで共有すると、着目ペア固有の特徴を取りこぼす。fidelityで勝ちながら真の係数再現率で負けるのは、Lassoの係数が縮小・相関特徴間で恣意的に選ばれるのに対し、共有集合上のRidge再フィットは偏りが小さいため。
  - **研究ストーリーの組み替え（2026-09-05、ユーザー承認済み・確定方針）**：Fisherは提案から降ろし分析章に位置づけ、Contrastive LIMEを提案手法とする。降ろした後「分析だけでなく新しい手法を出す」ため、Contrastiveの改良案を4つ試した（詳細は`docs/EXPERIMENT_LOG.md`フェーズ8〜11）。
    - **提案A「Multi-task Contrastive LIME」（全クラス同時のgroup lasso、循環整合性が構成上ゼロ）は不採用**。真の係数復元・fidelity・安定性の全軸で明確に劣る（例：真top-K再現率−0.21、fidelity−0.04）。**敗因の説明は訂正済み**：フェーズ7も全クラスの情報を共有しており「着目ペアだけの共有 vs 全クラス共有」という対比は誤りだった。正しい違いは「独立最適化の事後集約（フェーズ7）vs 結合最適化（提案A）」（詳細はEXPERIMENT_LOG.mdフェーズ8）。
    - **提案B「対比認識カーネル」**（2クラスが拮抗する局所領域を重視する摂動重み）：全体・穏やかな領域のfidelityが全9セルで有意に改善（+0.005〜+0.007）、極端領域は無風、安定性も6/9セルで改善。**ただしこのfidelityは学習に使った摂動`Z`上のin-sample測定であり、held-out再検証が必要**（下記参照）。
    - **提案C「OVO local logistic」**（Contrastiveと同じ$\log(p_{c_1}/p_{c_2})$目的変数を、Ridgeではなくソフトラベル交差エントロピーで回帰）：Bとほぼ同じ改善パターンだが、**安定性の改善がBより一貫**（9/9セルで有意 vs Bの6/9）。真の係数復元はリッジよりごく僅かに劣る（天井効果、0.998 vs 0.999）。**注記（訂正）**：損失自体は有界ではない（有界なのは勾配$\sigma(s)-q$）。RF側fidelityも同じくin-sample測定。
    - **B+C組み合わせ**：in-sample測定ではfidelityが素直に積み上がる（standard比+0.007、9/9有意）が安定性は打ち消し合う、という結果だった。**held-out（独立に引き直した摂動）で再検証した結果は下記「held-out再検証」を参照——in-sample版の数値は参考値として残すが、結論はheld-out版を優先する**。
    - **フェーズ12（2026-09-06、中心的な問いの直接検証、複雑さを揃えた対照実験で訂正済み）**：`run_ovo_vs_ovr_experiment.py`で、OVR-union（$c_1,c_2$独立top-K Lasso、係数差の台＝和集合）・Contrastive（K個に疎化）・Logistic/提案C（新規`fit_ovo_logistic_lasso`でK個に疎化）を、held-out摂動でのペア符号一致率で比較。**当初の結果**（OVR-unionが指定Kの1.3〜1.4倍の特徴を使う有利な条件）ではOVOとほぼ同点だったが、**OVR側を各クラス$\lceil K/2\rceil$個に絞った対照実験**（実際の複雑さはKの0.6〜0.8倍、不利側）では**Contrastive・Cの両方がOVR-unionに明確に勝つ**（それぞれ+0.045・18/18有意、+0.041・14/18有意）。正確に$K$へ一致させた版は未実装だが、有利側で同点・不利側で明確勝利という2点で挟まれているため、**複雑さを公平に揃えればOVOが勝つ可能性が高い**——「中心的な問いは支持されなかった」という当初の結論は撤回する。**ただし疎な設定でのC vs Contrastiveでは、依然としてCがわずかに劣る**（フェーズ11.5の密な設定とは逆の序列、3/18で有意）。「Cを主軸とする」という方針は密な全特徴フィットの場合に限定されることに変わりはない。
    - **held-out再検証（2026-09-05、指摘を受けて追加、確定結果）**：`run_combined_bc_experiment.py`に独立摂動`Z_test`での評価を追加して再検証（詳細はEXPERIMENT_LOG.mdフェーズ11.5）。**(i) B・Cそれぞれ単体のfidelity改善は規模が縮小しつつも生き残る**（standard比、全体fidelity+0.003〜0.005・4〜6/9セルで有意、in-sampleの9/9からは縮小）。**(ii) 「B+Cで積み上がる」というin-sample結論は再現されなかった**——combinedとkernel単体・logistic単体の差は0〜1/9セルしか有意にならない。in-sample版の上乗せは学習サンプルへの適合度の見かけ上の差だった可能性が高く、撤回する。(iii) 極端領域fidelityはtrain・test問わず一貫して無風。(iv) 方向の安定性（resamplingベースで元々in-sampleの問題を受けない）は変わらず：logistic単体が最も一貫（8/9）、combinedはlogistic単体より悪化（6/9有意）。
    - **採用方針（held-out再検証を反映して確定）**：提案手法はContrastive LIME。改良は**提案C（OVO logistic）単体を主軸とする**——held-outでもfidelity・安定性の改善が残る唯一の案。**提案Bの追加併用は積極的には推奨しない**（held-outでの上乗せがほぼ確認できず、安定性をわずかに損なう可能性がある）。一貫性が必要な場合は共有支持集合（上記(a)〜(c)、ただしこちらもin-sample測定のまま未検証）を選択肢として提示。Fisherは「この設定でlog-ratio係数の復元に劣った」という機構レベルの結果を示した分析章（一般的な限界の証明ではない）。
  - **グラウンドトゥルース検証（Rahnama et al. 2024型、2026-09-03追加）**：黒箱をRandomForestから多項ロジスティック回帰に差し替え（＝真の対数オッズ係数$\theta_{c^*}-\theta_{c'}$が既知になる）、各手法の推定係数と真の係数のSpearman順位相関を測定。これは今までの「fidelity（局所再現性）」とは質的に異なる軸で、「正しい特徴に重みを置けているか」を直接検証する。**結果はContrastiveの最も明確な勝ちどころになった**：全グリッド平均でContrastive ρ=0.999、Fisher(soft) ρ=0.983、OVR ρ=0.976、Fisher(hard) ρ=0.953という順位が、ほぼ全ペア・全9セルで統計的に確定した（Contrastive vs 他3手法：全9セルで有意にContrastiveが優位、p≦0.0003）。理論的にも整合する：Contrastiveは対数オッズを直接回帰するため、黒箱が対数オッズについて線形（多項ロジスティック回帰）である限り真の係数をほぼ完璧に復元できる。OVRは生の確率（softmaxで非線形）を回帰するため一致度が下がる。Fisher(hard)は最下位で、これまでの実験（stability・極端領域fidelity）で見えていた「ハードラベルのサンプル飢餓」問題が別角度から再確認された。

## 主要な構成

- `src/perturbation.py`: one-vs-rest LIMEとFisher LIMEに共通の摂動サンプリング（LIMEのデフォルト方式：Gaussianノイズ×特徴量標準偏差、指数カーネルで近接度重み付け）。両手法を公平に比較するため、この摂動生成ステップだけは完全に共有する設計。
- `src/surrogates.py`: サロゲートフィッティング関数群。
  - `fit_onevsrest`: クラスごとに独立な重み付きRidge回帰（全特徴量使用、intercept込み）。
  - `fit_onevsrest_lasso`: クラスごとに独立な重み付きLasso選択（`lasso_path`スタイル、二分探索でK個以上の非ゼロ係数を持つ最疎解を求めて上位K個を採用）。真の特徴量部分集合選択を再現するため、feature overlap実験で使用。
  - `fit_fisher`: 3クラス（以上）共通のプールされたクラス内散布行列S_Wを使うFisher LDAサロゲート（ハードラベル版）。shrinkage正則化あり。ペア方向`v(X,Y)=S_W^{-1}(μ_X-μ_Y)`と one-vs-rest形式の`onevsrest_direction(c)=S_W^{-1}(μ_c-μ_¬c)`を計算するヘルパーを返す。
  - `shared_support_fisher_soft` / `shared_support_ridge` / `fit_contrastive_on_support`（2026-09-05、提案）: 二段構成サロゲート。Stage 1で全クラス共通のtop-K特徴集合を1つ選び（Fisher soft方向の集約、または対照としてOVR Ridge係数の集約）、Stage 2でその集合に制限したContrastive log-odds Ridgeを各ペアにフィットする。ペア横断のfeature overlapは構成上1。
  - `fit_fisher_soft`: ソフトラベル版。`π_i・f_c(z_i)`を重みとして使い、argmaxによるハードラベル化を行わない。クラスが局所近傍から丸ごと欠落する問題を解消するが、feature overlap自体は改善しないことがある（トレードオフ、詳細は`docs/RECENT_WORK.md`）。
  - `fit_contrastive`: 「Contrastive LIME」。$\log((p_{c1}+\varepsilon)/(p_{c2}+\varepsilon))$を目的変数にした重み付きRidge回帰（ペアごとに独立フィット、Fisherの共有S_Wは使わない）。
  - `fit_contrastive_lasso`: 同じ目的変数でのLasso選択版（feature overlap実験用）。
  - `top_k_indices`: 上位K個（絶対値）の特徴量インデックス集合を返すヘルパー。
- `src/metrics.py`: consistency・stability指標。
  - `sum_to_one_deviation` / `sum_to_one_deviation_topk`: 全特徴量時と top-K切り詰め後のsum-to-one逸脱。
  - `mean_pairwise_feature_overlap`: LIMEtreeの主張（共通構造の有無）を操作化した、クラス間top-K特徴量集合の平均Jaccard重なり。
  - `total_variance`: stability指標（ペア方向ベクトルの分散のtrace）。**警告**：Fisherとone-vs-restの出力ベクトルは尺度が異なるため、これを直接比較するのは誤り（詳細は`docs/RECENT_WORK.md`）。
  - `total_variance_normalized`: 尺度不変のstability指標（単位ベクトルに正規化してから分散を取る）。**手法間の比較には必ずこちらを使う**。
  - `mean_norm`: ベクトルの平均大きさ（尺度差を確認するための参考情報）。
  - `transitivity_violation_rate`: **理論的に常に0になるため実験では未使用**。docstringに理由を明記した上でコードのみ残置。
- `src/check_identities.py`: データに依存しない代数的な恒等式チェック（統計的検定ではない）。(1) OVRの係数差＝$p_c-p_d$の直接Ridge回帰と数式上同一であること、(2) 密なContrastiveの循環整合性$\hat\beta_{ab}+\hat\beta_{bc}=\hat\beta_{ac}$、(3) `fit_contrastive`と`fit_ovo_logistic`のepsilon平滑化が同じ量を表していること、を機械精度で検証する。
- `src/stats_utils.py`: 実験ドライバ共通の統計ヘルパー。`bootstrap_ci`（パーセンタイル・ブートストラップ信頼区間）、`paired_wilcoxon`（対応ありWilcoxon符号順位検定＋matched-pairs rank-biserial効果量）、`holm_bonferroni`（多重比較補正）、`compare_methods`（グリッドセルごとにseedレベル平均を独立サンプルとして扱い、これらを組み合わせて統計比較表を作る高水準関数）。擬似反復（同一データセット内の複数インスタンスを独立サンプル扱いすること）を避ける設計上の理由はモジュールのdocstringに詳しく記載。
- `src/run_experiment.py`: 次元数×クラス数×Kのグリッドで one-vs-rest / Fisher(hard) を比較する実験ドライバ。各グリッドセルにつき独立したデータセット抽選を`N_DATASET_SEEDS=20`回繰り返す。完走済み、生データは`results/experiment_results.csv`、統計比較は`results/experiment_stats.csv`に出力。
- `src/investigate_reversal.py`: feature overlapでFisherの優位性が縮小・逆転する条件（高クラス数、n_classes=6〜7）の原因を切り分ける診断スクリプト。ハード版・ソフト版Fisherを同時比較する。**まだ`N_DATASET_SEEDS`方式の統計的厳密化の対象外**（単一シードのアドホック診断のまま）。
- `src/fidelity.py`: 忠実性（fidelity）評価用の確率変換・損失関数。`onevsrest_predict_proba`（Ridge出力のクリップ＋正規化）、`fisher_predict_proba`（LDA確率モデルによる擬似確率、`LinearDiscriminantAnalysis.predict_proba`と同じ考え方）、`weighted_hellinger_loss`（SLISEMAP Eq.11と同じ二乗Hellinger距離）。
- `src/run_fidelity_experiment.py`: 次元数×クラス数グリッドでone-vs-rest / Fisher(hard) / Fisher(soft)の忠実性（Hellinger損失）を比較する実験ドライバ。`N_DATASET_SEEDS=20`で反復。結果は`results/fidelity_results.csv`、統計比較は`results/fidelity_stats.csv`。
- `src/run_contrastive_experiment.py`: one-vs-rest / Fisher(hard) / Contrastiveの3手法を、fidelity（ペア符号一致率）・stability（正規化）・feature overlapの3指標で同時比較するグリッド実験。`N_DATASET_SEEDS=20`で反復。結果は`results/contrastive_results.csv`、統計比較は`results/contrastive_stats.csv`。
- `src/run_extreme_regime_experiment.py`: 同じ局所近傍を「競合2クラスの確率が両方とも極端（一方が閾値未満）」と「穏やか」に分割し、fidelityを領域別に比較するグリッド実験。`N_DATASET_SEEDS=20`で反復。結果は`results/extreme_regime_results.csv`、統計比較は`results/extreme_regime_stats.csv`。
- `src/diagnose_fisher_direction.py`: Fisher方向$S_W^{-1}(\mu_c-\mu_d)$がなぜ回帰より真の係数を復元できないかを部品分解する診断（重心差のみ／pooled $S_W$／ペア限定$S_W$／対角$S_W$／連続応答版、hard/soft）。黒箱はロジスティック回帰。結果は`results/diagnose_fisher_direction_{results,stats}.csv`。
- `src/run_shared_support_experiment.py`: 提案（二段構成：Fisherで共有特徴集合を選び、Contrastive回帰で係数を出す）の評価。対照は同じ二段構成でStage 1をOVR Ridge係数集約にした版と、per-pair Contrastive Lasso。黒箱はロジスティック（真のtop-K再現率・Spearman）とRF（ペア符号fidelity・特徴集合安定性・方向安定性・ペア横断overlap）。結果は`results/shared_support_{logistic,rf}_{results,stats}.csv`。
- `src/run_groundtruth_experiment.py`: Rahnama et al. (2024)型のグラウンドトゥルース検証。黒箱を`LogisticRegression(multinomial)`に差し替え、真の対数オッズ係数$\theta_{c^*}-\theta_{c'}$と各手法（OVR / Fisher hard / Fisher soft / Contrastive）の推定係数のSpearman順位相関（`metrics.pairwise_coef_spearman`）を測る。`N_DATASET_SEEDS=20`で反復。結果は`results/groundtruth_results.csv`、統計比較は`results/groundtruth_stats.csv`。
- `docs/OVO_LIME_METHODS.md`: OVR、pairwise probability、pairwise log-odds、OVO Logistic-LIME、Contrastive LIME、OVO Fisher-LIME、共同学習の定式化と評価案。実装状況表あり。
- `docs/RELATED_WORK.md`: 多クラスLIMEと対比的局所説明に関する査読済み文献、各研究との重なり、安全な新規性の位置づけ、比較実験への示唆。
- `docs/EXPERIMENT_LOG.md`: **今までの全実験の詳細な記録**（フェーズ1〜11、各実験の目的・方法・数値・結論を`results/*.csv`から再集計して記載）。個々の実験の詳細を確認する際はまずここを見る。PROJECT_SUMMARY.md/RECENT_WORK.mdは要約、EXPERIMENT_LOG.mdは一次記録という役割分担。
- `src/diagnose_fisher_direction.py`: Fisher方向$S_W^{-1}(\mu_c-\mu_d)$を部品分解（重心差のみ／全クラスプール／ペア限定／対角のみ／連続応答版）し、真の係数（LogisticRegression黒箱）との順位相関で機構レベルの原因を特定する診断スクリプト。
- `src/run_shared_support_experiment.py`: 「共有支持集合＋Contrastive再フィット」二段構成（`shared_support_fisher_soft`/`shared_support_ridge`/`fit_contrastive_on_support`、いずれも`src/surrogates.py`）と、提案A「Multi-task Contrastive LIME」（`fit_joint_contrastive`、全クラス同時のgroup lasso）を、per-pair Contrastive Lassoと比較する実験ドライバ。両黒箱（RF/LogisticRegression）対応。
- `src/run_pair_kernel_experiment.py`: 提案B「対比認識カーネル」（`contest_weights`、$\pi_i\cdot(4q_i(1-q_i)+\text{floor})$）の評価。
- `src/run_logistic_target_experiment.py`: 提案C「OVO local logistic」（`fit_ovo_logistic`、ソフトラベル交差エントロピー）の評価。
- `src/run_ovo_vs_ovr_experiment.py`: 中心的な問い（同じ表示特徴数ならOVOの方が忠実か）をheld-out摂動で直接検証する実験。OVR-union（独立top-K Lassoの係数差、複雑さ＝和集合として正直に記録）・Contrastive（K個に疎化）・Logistic（`fit_ovo_logistic_lasso`でK個に疎化）を比較。結果は`results/ovo_vs_ovr_{results,stats}.csv`。
- `src/run_combined_bc_experiment.py`: 提案B・Cが積み上がるかの検証（standard/kernel/logistic/combinedの4水準）。fidelity/extreme/moderateは`_train`（学習に使った摂動上、in-sample、参考値）と`_test`（独立に引き直した摂動上、held-out、**こちらが正**）の両方を出力する。
- `.venv/`: Python仮想環境（`.gitignore`で除外、コミット対象外）。

## セットアップと実行方法

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 src/run_experiment.py         # メイングリッド実験。結果は results/ にCSV出力
python3 src/investigate_reversal.py   # 高クラス数での逆転を調べる診断実験
```

必要パッケージ: numpy, scipy, scikit-learn, pandas, lime, matplotlib（`requirements.txt`参照）。

## 重要な設計判断

- **one-vs-rest LIMEとFisher LIMEは、同一の摂動サンプル・同一の近接度重みを使う**（`src/perturbation.py`で共有）。これは元の問題提起の前提条件（「摂動データは同じサンプリングとする」）を実験でも忠実に再現するため。
- one-vs-rest側の黒箱への問い合わせターゲットは各クラスの確率 `predict_proba` の各列。Fisher側は主にハードラベル（`argmax`）でクラスを割り当ててから重心・散布行列を計算するが、ソフトラベル版（`fit_fisher_soft`）も実装済み。どちらが良いかは条件依存（`docs/RECENT_WORK.md`参照）。
- Fisher側のS_Wは正則化のため `S_W + ε·(trace(S_W)/d)·I` のshrinkageを適用している（`shrinkage`パラメータ、デフォルト1e-3）。
- feature overlap実験（consistency検証）では、one-vs-rest側は真のLasso選択（`fit_onevsrest_lasso`）を使う。単純なRidge事後top-K切り詰めでは差が出なかったため（詳細は`docs/RECENT_WORK.md`）。
- 合成データは`n_redundant = n_features - max(3, n_classes)`で、相関の強い冗長特徴量を意図的に含める。Lassoの特徴量選択不安定性（相関特徴量群からの恣意的な選択）を再現するため。

## データと環境依存

- 合成データ（`sklearn.datasets.make_classification`）を使用。実データセットは今のところ使用していない。
- 黒箱モデルは`RandomForestClassifier`（非線形決定境界を作るため）。
- 秘密情報・環境依存の外部サービスなし。

## 既知の制約・リスク

- **手法間で出力ベクトルの生の大きさ・分散を直接比較しない**。Fisherの$v=S_W^{-1}(\mu_X-\mu_Y)$には自然な尺度がなく、one-vs-restの回帰係数と直接比較すると尺度差だけで数十〜数百倍の見かけの差が生まれる。比較する際は必ず正規化（`total_variance_normalized`など）を使うか、両手法とも適切な単位（実際の確率値など）に変換してから比較すること。
- `transitivity_violation_rate`（推移律違反率）という指標は理論的に常に0になる無意味な指標であることが判明。使用しない（docstringに理由明記済み）。
- sum-to-one（$\hat P(A)+\hat P(B)+\hat P(C)=1$）は、one-vs-rest側が全特徴量を使った同一設計の回帰であれば厳密に成立してしまう。崩れを見たい場合はtop-K切り詰め（`sum_to_one_deviation_topk`）で検証する。
- **Fisher（ハードラベル版）はクラス数が多い、または局所サンプル数が少ない場合、局所近傍にそのクラスのサンプルが1つも現れず、そのクラスの説明が丸ごと欠落することがある**（`min_class_count_avg`が0近くになる事例を確認済み）。ソフトラベル版はこの欠落を解消するが、feature overlap自体はむしろ悪化する条件が多い。単純にどちらかを常に推奨できる状態ではない。
- Fisher LDAのS_Wは次元が高くローカルサンプル数が少ない場合に特異・悪条件になりうる（shrinkage正則化で緩和しているが、パラメータの妥当性は未検証）。

## 今後の大きな課題

- **最優先**：提案Cが密な設定（フェーズ11.5）と疎な設定（フェーズ12）で逆の結論になる原因を調査する（複雑さを揃えた対照実験でも解消しなかった、Contrastive vs Cの序列自体の問題）。ソフトラベル交差エントロピーのL1正則化パスの挙動を診断する必要がある。修論の主張をどちらの設定に基づいて組み立てるか（またはその使い分けを明示するか）を決める前提として重要。
- OVR-unionの複雑さを正確に$K$へ一致させた比較を実施する（フェーズ12でK/2版により「有利側・不利側で挟む」ところまでは実施済み、正確な一致は未実施）。
- 共通特徴・ペア固有特徴・無関係特徴を明示的に作る合成データ生成器を実装し、フェーズ12で保留した仮説2（共通特徴の回避）・仮説3（競合クラス変更への追随）を検証する。
- **フェーズ1〜8のfidelity測定（`run_experiment.py`, `run_contrastive_experiment.py`, `run_fidelity_experiment.py`, `run_extreme_regime_experiment.py`, `run_shared_support_experiment.py`のRF側）を、フェーズ11.5と同じ方式（独立held-out摂動）で再検証する**。B/C/combinedで再検証したところ、in-sampleの効果量・有意性が有意に縮小することが判明したため、優先度が高い。
- consistencyの主張を、実測で裏付けられる正確な形（LIMEtreeの「共通構造の有無」の定義に基づく、条件付きの主張）に修論の記述を修正する。
- ハード版・ソフト版Fisherのトレードオフを理論的に説明する（重心間距離・S_Bの直接比較など）。ハイブリッド案（局所サンプルが少ないクラスだけソフトにフォールバック）の検討。
- ソフト版の正規化stabilityを、今回整備した統計的枠組み（`N_DATASET_SEEDS`反復＋`stats_utils.py`）でフルグリッド再検証し、恒久的なスクリプトとして組み込む（現状3セルのアドホック検証のみ）。
- `src/investigate_reversal.py`（n_classes=6〜7での「逆転」診断）を同じ統計的枠組みで再検証する。今回の`run_experiment.py`再検証（n_classes 3〜5では「逆転」ではなく「優位性の消失」）との整合性を確認する必要がある。
- feature overlapの「クラス間 vs ペア間」という比較単位の不一致を解消し、Contrastive・Fisher・one-vs-restを公平に再比較する。
- one-vs-rest, Fisher(hard/soft), Contrastiveの4手法×fidelity・stability・feature overlap・sum-to-one・極端領域fidelity・グラウンドトゥルース復元の、統計的に裏付けられた結果を統合し、修論の主張として文章化する。「Fisherが優れている」という単純な主張ではなく、条件付き・トレードオフとして誠実に書く必要がある。グラウンドトゥルース検証はContrastiveにとって最も明確に勝てる指標なので、修論の主張の中心に据えることを検討する。
- インスタンス選択（マージン最小の8点のみ）が結果を偏らせていないか検証する（今回のスコープ外、ユーザーの優先度確認により統計的厳密性のみ対応）。
- 実データセットでの再現性確認。
