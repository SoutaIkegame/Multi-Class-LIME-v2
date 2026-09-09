# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-09（Codex Desktop、自宅Mac）
- 作業環境: ローカル実行環境（Codex Desktop）
- ブランチ: `main`
- 基準コミット: `5d62c99`

## 今回の目的

OVRと上位2クラスを説明する3方式を、同じ表示特徴数、held-out摂動、複数の黒箱で比較し、発表で使える結論と図を得る。

## 指摘4点と対応

1. **OVR/OVOは近似する量が違うのであって、目的変数が同じで損失だけ違うわけではない**：`docs/OVO_LIME_METHODS.md`・`docs/EXPERIMENT_LOG.md`の表現を訂正。
2. **「直接回帰だから優れる」という説明は不正確**：Ridge回帰は目的変数に対する線形演算子なので、同一設計・重み・正則化なら「OVRの係数差」と「$p_c-p_d$の直接回帰」は数式上同一の推定量。`src/check_identities.py`（新規）で機械精度まで確認した（`check_ovr_difference_identity`）。実際の変更点は目的変数の変換（確率差→対数比）であり、「直接 vs 引き算」ではない。
3. **循環整合性は「何も強制しない」のではなく、密なRidge版では数式上厳密に成立する**：`check_identities.py`の`check_cycle_consistency`で確認。Lasso版（ペアごとに特徴選択が変わる）でのみ崩れうる。
4. **`fit_contrastive`と`fit_ovo_logistic`のepsilon平滑化が不整合だった**：`fit_ovo_logistic`のqを$(p_{c_1}+\varepsilon)/(p_{c_1}+p_{c_2}+2\varepsilon)$に修正し、`logit(q)`が`fit_contrastive`の対数比と厳密に一致するようにした。`check_identities.py`の`check_smoothing_consistency`で確認。

**さらに「中心的な問いを精緻化してほしい」という依頼を受け**、「特定の競合クラスとの違いを説明するとき、OVRより少ない特徴で忠実に説明できるか」を研究の中心的な問いとして確定し、`docs/OVO_LIME_METHODS.md`・`docs/PROJECT_SUMMARY.md`に明記した。

## 新規実装：フェーズ12（中心的な問いの直接検証）

**2026-09-08追記**：以下の旧結果は、特徴選択後の最終サロゲート再フィットが欠けていたため、後段の「追加実験：2クラス選択後の通常LIME」で再検証・置換した。発表には後段の値を使う。

この問いを実際に検証するため、`src/run_ovo_vs_ovr_experiment.py`を新規実装した。

- `fit_ovo_logistic_lasso`（`src/surrogates.py`新規）：`fit_ovo_logistic`のL1正則化版。L1ロジスティック回帰（`liblinear`）でK個ちょうどに疎化する二分探索（`_sparsest_logistic_l1_with_at_least_k`）。
- OVR-union・Contrastive（Lasso）・Logistic（新規Lasso版）を、**held-out摂動**（最初から独立摂動で評価、フェーズ11.5の教訓を反映）でのペア符号一致率で比較。

### 結果（重要、フェーズ11.5と一部矛盾する）

- **Contrastive vs OVR-union**：ほぼ完全な同点（差−0.0007、0/18有意）。OVR-unionは実際には指定Kの1.3〜1.4倍の特徴を使っており（2クラス独立選択の和集合のため）、複雑さで有利な条件でもContrastiveに負けていない。
- **Logistic（提案C）vs OVR-union・Contrastive**：**Cがわずかに劣る**（3/18・2/18で有意、全て同じ方向）。
- **これはフェーズ11.5（密な全特徴フィット）でCがContrastiveより有意に優れていた結果と表面的に矛盾する**。整合的な解釈：Cの優位性は密な設定に限定され、L1による疎化を経由すると消える、あるいは逆転する（原因未解明、最優先の持ち越し課題）。

「同じ表示特徴数ならOVOの方が忠実」という中心的な問いは、疎な設定では支持されなかった、というのが当初の結論だった。

**追記（同日中、ユーザーからの疑問を受けて訂正）**：この結果はOVR-unionが指定Kの1.3〜1.4倍の特徴を使う有利な条件での比較だった。`ovr_union_half`（各クラス$\lceil K/2\rceil$個、実際の複雑さはKの0.6〜0.8倍・不利側）を追加した対照実験では、**Contrastive・Cの両方がOVR-unionに明確に勝つ**（それぞれ18/18・14/18で有意）。正確に$K$に一致させたわけではなく有利側・不利側で挟んだだけだが、複雑さを公平に揃えればOVOが勝つ可能性が高いと判断し、「支持されなかった」という結論は撤回した。ただし疎な設定でのC vs Contrastiveの序列（Cがわずかに劣る）自体は変わっておらず、密/疎での逆転は未解決のまま。

## 実施した変更と主要な変更ファイル

1. `src/check_identities.py`（新規）：3つの恒等式チェック（OVR差分、循環整合性、epsilon平滑化）。
2. `src/surrogates.py`：`fit_ovo_logistic`のepsilon修正、`fit_contrastive`・`fit_ovo_logistic`周辺のdocstring訂正、`fit_ovo_logistic_lasso`・`_sparsest_logistic_l1_with_at_least_k`追加。
3. `src/run_ovo_vs_ovr_experiment.py`（新規）：フェーズ12の実験。フルグリッド実行済み。
4. `docs/EXPERIMENT_LOG.md`：「0.1 訂正記録」新設、フェーズ4の訂正、フェーズ12追加、以降のセクション番号を繰り下げ。
5. `docs/OVO_LIME_METHODS.md`：中心的な問いの確定、訂正、OVOの価値の一般化された説明（共通成分の相殺）を追記。
6. `docs/PROJECT_SUMMARY.md`：プロジェクトの目的を現状（Contrastive+C提案、Fisher分析章）に合わせて更新、フェーズ12の結果を追記。

## 未完了・既知の問題・未検証事項

- **最優先**：提案Cの密/疎での逆転の残存原因調査。選択後再フィットの欠落は修正したが、支持集合・正則化強度による差が残る。
- OVR-exactによるK一致と、共通／競合／無関係特徴を明示した実験は2026-09-09に完了した。別のK配分規則と、競合クラスが変わる場合の追随性は未検証。
- 前回からの持ち越し：フェーズ1〜8のfidelityのheld-out化、ソフト版Fisherのstability、`investigate_reversal.py`の統計化、実データ未検証。
- スライド（`draft_slides.pptx`）は前回・今回の訂正内容に未反映のまま。

## 追加実験：2クラス選択後の通常LIME（2026-09-08）

上位2クラス$c_1,c_2$を固定し、$q=p_{c_1}/(p_{c_1}+p_{c_2})$へ通常LIMEを適用する`fit_pairwise_lime`／`fit_pairwise_lime_lasso`を追加した。検証中、既存のフェーズ12が特徴選択用Lasso/L1の縮小係数を最終説明にも直接使い、通常LIMEの選択後再フィットを欠いていたことが判明した。このため`run_ovo_vs_ovr_experiment.py`内でOVR、2クラス通常LIME、Contrastive、OVO Logisticをすべてselect-then-refitへ統一してフル実験を再実行した。

- 条件：RF黒箱、合成データ、特徴数8/14/20、クラス数3/4/5、K比0.25/0.5、各セル20独立seed、各seed 8説明点、学習・評価摂動各300。
- held-outペア符号忠実度：OVR-union 0.7920（7.35特徴）、2クラス通常LIME 0.7914（5.33特徴）、Contrastive 0.7889、OVO Logistic 0.7818、OVR-half 0.7700（4.19特徴）。
- 2クラス通常LIME vs Contrastive：+0.00243、全体$p=3.8\times10^{-6}$、Holm後6/18セルで通常LIME優位。
- 2クラス通常LIME vs OVO Logistic：+0.00957、全体$p=1.9\times10^{-6}$、Holm後18/18セルで通常LIME優位。
- 2クラス通常LIME vs OVR-union：−0.00067、$p=0.45$で同点。平均約27%少ない表示特徴数で同等だった。
- weighted Brier：2クラス通常LIME 0.01559、Contrastive 0.01579、OVO Logistic 0.01681。通常LIMEが最小。

**結論更新**：RFでは、2クラスの確率の合計を1にして通常LIMEを適用するだけで主要な利得が得られ、logit変換およびcross-entropyによる追加改善は確認できなかった。その後の追加実験では、Contrastiveは線形softmaxの忠実度・Brier・真係数復元で最良だったが、非線形softmaxとRFでは2クラス通常LIMEを下回った。旧フェーズ12の+0.0453/+0.0405は再フィット欠落の影響を含むため最新結論には使わない。

変更・生成物：`src/surrogates.py`、`src/run_ovo_vs_ovr_experiment.py`、`src/plot_pairwise_lime_baseline.py`、`results/ovo_vs_ovr_{results,stats}.csv`、`results/pairwise_lime_baseline_{summary,overall_stats}.csv`、`results/pairwise_lime_baseline_comparison.png`。

確認：`python3 -m py_compile src/surrogates.py src/run_ovo_vs_ovr_experiment.py`、`python3 src/run_ovo_vs_ovr_experiment.py`（全9セル完走）、`python3 src/plot_pairwise_lime_baseline.py`。

## exact-K OVRと黒箱構造の追加実験（2026-09-09）

2本のOVRが選んだ候補和集合を係数差で順位付けして全体でちょうどK特徴へ絞り、同じ集合上で2本を再フィットするOVR-exactを追加した。RFでは、held-out符号忠実度が2クラス通常LIME 0.7914、Contrastive 0.7889、OVR-exact 0.7873、OVO Logistic 0.7818。2クラス通常LIMEはOVR-exactより+0.00406（全体$p=5.7\times10^{-6}$、Holm後5/18セル）で、同じ特徴数でも優位が残った。

線形softmax、非線形NN＋softmax、RFの3黒箱を同じexact-K設計で比較した。

- 線形softmax：Contrastive 0.8655、2クラス通常LIME 0.8632、OVO Logistic 0.8560、OVR 0.8492。
- 非線形NN＋softmax：2クラス通常LIME 0.7840、Contrastive 0.7808、OVR 0.7763、OVO Logistic 0.7762。
- RF：2クラス通常LIME 0.7847、Contrastive 0.7819、OVR 0.7817、OVO Logistic 0.7745。

線形softmaxの真係数復元も再実行し、Spearman相関はContrastive 0.9993、OVO Logistic 0.9975、2クラス通常LIME 0.9872、OVR 0.9758だった。**softmax自体ではなく、2クラスのlogit差が入力に対して線形または局所線形に近いことがContrastiveの利点を活かす条件**である。

統計処理では、`compare_methods`のHolm補正が異なる手法ペアまで同じ検定族に含めていた問題を修正した。現在は各手法ペアについて全グリッドセルを1検定族として補正する。関連する`ovo_vs_ovr_stats.csv`と`groundtruth_stats.csv`は修正版で再生成済み。

変更・生成物：`src/stats_utils.py`、`src/run_ovo_vs_ovr_experiment.py`、`src/run_groundtruth_experiment.py`、`src/run_blackbox_comparison_experiment.py`、`src/plot_blackbox_comparison.py`、`results/blackbox_comparison_{results,stats,summary}.csv`、`results/blackbox_comparison.png`。

確認：3つのフル実験は完走し、全結果が有限、全exact-K方式の複雑さがKと一致。`src/check_identities.py`の3恒等式も全て合格。黒箱比較図は目視確認済み。

## 特徴の役割を既知にした解釈性実験（2026-09-09）

線形softmax BBの特徴を、A/Bの係数が反対の競合特徴3個、A/Bの係数が同じ共通特徴3個、他クラス専用・無関係特徴に分けた。共通特徴の強度0/1/3/5、クラス数3/5、K=3、20seedで完走した。

共通強度5では、競合特徴recallが通常のtop-class LIME 0.780、OVR-exact 0.807まで低下した一方、2クラス通常LIMEとContrastiveは1.000、OVO Logisticは0.999だった。OVR-exactは表示特徴の19.3%をA/B共通特徴に使い、ペア型3方式はほぼ0%だった。held-out忠実度もOVR 0.9088に対して、2クラス通常LIME 0.9827、Contrastive 0.9927、OVO Logistic 0.9894だった。

この結果は「人が必ず理解しやすい」ことの直接評価ではないが、2クラス説明が表示枠をA/Bの違いに関係する特徴へ集中できることを、既知の真値に対するrecallとして定量化した。図`results/feature_role_comparison.png`は目視確認済み。

## 計算時間の再測定（2026-09-09）

全方式をselect-then-refitへ統一し、OVR-exact、2クラス通常LIME、Contrastive、OVO Logisticを特徴数8/14/20、クラス数3〜10、10seedで再測定した。全条件平均は順に41.44、6.54、6.52、15.89 ms/説明で、OVRは通常LIME／Contrastiveの約6.3倍、OVO Logisticの約2.6倍だった。クラス数3→10でOVRは20.4→63.5 msへ増えたが、1ペアだけを作る3方式はほぼ横ばいだった。次元8→20ではOVO Logisticが11.94→20.03 ms、他の方式の増加は小さかった。

`results/timing_scaling.png`に次元数推移・クラス数推移を並べ、目視確認した。これはBB推論と摂動生成を除いたサロゲートフィット時間であり、全クラス対を作る場合のOVO時間ではない。

発表用の比較図4枚（RF比較、黒箱構造別、特徴役割、計算時間）は、タイトル・軸・凡例・注記を日本語へ統一し、游ゴシックで再生成して文字化けとレイアウトを目視確認した。

`docs/DISCUSSION.md`を追加し、2クラス選択の効果、解釈性、対数比が有効な条件、OVO Logisticの位置づけ、計算時間、新規性、限界、発表で使う考察文を最新実験に基づいて整理した。

## 中間発表デッキへの結果セクション追加（2026-09-09）

`docs/MIDTERM_PRESENTATION_2026-09-12.pptx`を変更せずに複製し、初版`docs/MIDTERM_PRESENTATION_WITH_RESULTS_2026-09-09.pptx`を作成後、ユーザーの指摘を反映した現行版`docs/MIDTERM_PRESENTATION_WITH_RESULTS_V2_2026-09-09.pptx`を生成した。現行版は既存10枚の後ろへ次の9枚を追加した。

1. 評価設計：OVRと上位2クラスを対象にした3方式、4つの評価指標。
2. 実験設定：データ、次元数、クラス数、K、BB、説明点、摂動、反復、サロゲートのハイパーパラメータ。
3. RF主実験：OVR-exactと2クラス方式のheld-outペア符号忠実度・Brier。
4. BB別比較：線形softmax、非線形NN＋softmax、RF。
5. 解釈性：競合特徴再現率と共通特徴の混入。
6. 計算時間：次元数8/14/20、クラス数3〜10。
7. 考察1：OVRと2クラス説明が答える質問の違い、共通証拠の相殺、表示予算。
8. 考察2：2クラス通常LIME、Contrastive、OVO Logisticの条件付きの使い分け。
9. 結果のまとめ：支持されたこと、条件付きの結果、未検証事項。

V2ではユーザー指定によりグラフ4枚を`results/*.png`の画像として配置した。各追加スライドの下部に「このスライドの主張」を一文で明示し、結果スライドには数値の読み方、考察スライドには原因解釈と主張の限界を追加した。実験設定表は編集可能なネイティブ表のまま。全19枚を再描画して目視確認し、元の10枚は維持した。最終検証ではslide count 19、image graph 4、native table 1、パッケージ整合性、レイアウト、フォント、Artifact Tool再インポート、テンプレート継承がすべて合格した。既存slide 3のconnector-over-text警告1件は元ファイル由来で、追加スライドには検出事項なし。

## 次に行うこと

### 研究の位置づけ変更（2026-09-08、ユーザー決定）

- **多クラスBBの予測上位2クラスに絞り、その2クラスを区別する局所説明として研究を構成する。** $q_{c,d}=p_c/(p_c+p_d)$は2クラスの確率の合計を1にするための処理であり、独自名称を付けない。この処理は通常LIME、log-odds Ridge、local logisticなどに共通して利用できる。
- 研究全体をCLIMAX拡張とは呼ばない。CLIMAXのL-CLIMAX／CE-CLIMAXは、条件付け後のlog-odds Ridge／local logisticに最も近い先行研究・比較対象として扱い、対数オッズ回帰やcross entropy自体を新規性として主張しない。
- 中心的な問いは、OVR型のクラス別説明とpairwise条件付き説明では、特定の競合ペアを説明する際の忠実性・疎性・安定性・解釈がどう変わるか、とする。現在の実験では説明点の予測1位・2位を固定して比較する。
- 新規性候補は、OVR／2クラスを選んで通常LIME／Contrastive Ridge／OVO Logisticを、同一摂動・同一特徴予算・held-outデータで比較し、2クラスに絞る効果とリンク関数／損失の効果を分離する実証設計である。
- 二値分類では$q_{c,d}=p_c$となって既存の二値説明と一致するため、対象は3クラス以上と明記する。「初のcontrastive explanation」「独立に新しい損失を提案した」とは表現しない。
- 発表の関連研究は、(1) GANMEXから特定2クラスを比較するOne-vs-One説明の着想、(2) CLIMAXからlog-odds Ridge／local logisticという回帰対象・当てはめ方の着想、(3) 両者を多クラスBBの予測1位・2位へ適用する提案、という一本の流れにする。Pairwise Couplingは$q=p_c/(p_c+p_d)$の式の出典として添える。
- 「なぜ対数比か」は本編スライドから外し、時間がある場合の補足・質疑応答用スライドとする。
- 実データと、競合クラスが変わる場合の追随性は未実施なので、完了前は効果を断定しない。

### 発表準備の追加TODO（2026-09-08、ユーザー依頼）

- [x] softmaxを含むBBで追加比較した。線形softmax、非線形NN＋softmax、RFをexact-K・held-outで完走し、図も生成した。
- [x] **上位2クラスを選んでから通常のLIMEを適用するベースライン**を追加し、select-then-refitへ統一したフル実験を完了した。結果は上記「追加実験：2クラス選択後の通常LIME」を参照。
- [x] OVRの表示特徴数を厳密にKへ揃えた比較を追加した。
- [x] select-then-refit修正版と2クラス通常LIMEを含めて計算時間実験を再実行し、次元数・クラス数推移の図を生成した。
- [x] クラス共通特徴・A/B競合特徴・無関係特徴を明示した合成BBで、特徴選択とheld-out忠実度を評価し、図を生成した。
- [ ] 時間があれば、質疑応答用スライド「普通の比ではなく、なぜ対数比か」をユーザーが作成する。A/B交換で符号だけ反転する対称性、等確率で0、線形softmaxではスコア差と一致する点を説明する。正規化項の相殺は普通の比でも成立し、softmaxでも入力に対する線形性は一般には保証しない点を添える。本編ではなく補足スライドへ回す。

### 既存の持ち越し

1. 提案Cの密/疎逆転の残存原因を調査する。再フィット欠落は修正済みだが、支持集合・正則化強度の差が残る。
2. 中間発表デッキ全体の話す順序を確認し、必要なら結果セクションへの接続文を調整する。結果セクション自体は最新結論へ更新済み。
3. 時間があれば、対数比を使う理由の質疑応答用スライドを追加する。
