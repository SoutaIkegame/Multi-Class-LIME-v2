# Recent Work

<!--
この文書は直近作業の引き継ぎ専用です。原則として各push前に現在の内容へ置き換えます。
前回内容のうち恒久的に必要な情報は、削除する前にPROJECT_SUMMARY.mdへ反映してください。
-->

## 更新情報

- 更新日時: 2026-09-06（Claude Code on the web、リモートセッション）
- 作業環境: リモート実行環境（Claude Code）
- ブランチ: `claude/research-theme-evaluation-jkkx9c`（PR #5）
- 基準コミット: `c506744`

## 今回の目的

ユーザーから2回目の詳細な外部レビュー（Contrastive LIME・OVOアプローチの理論的説明に対する4点の指摘）を受けた。全て検証の上、コード修正・恒等式による検証・ドキュメント修正・新規実験の実装まで行った。

## 指摘4点と対応

1. **OVR/OVOは近似する量が違うのであって、目的変数が同じで損失だけ違うわけではない**：`docs/OVO_LIME_METHODS.md`・`docs/EXPERIMENT_LOG.md`の表現を訂正。
2. **「直接回帰だから優れる」という説明は不正確**：Ridge回帰は目的変数に対する線形演算子なので、同一設計・重み・正則化なら「OVRの係数差」と「$p_c-p_d$の直接回帰」は数式上同一の推定量。`src/check_identities.py`（新規）で機械精度まで確認した（`check_ovr_difference_identity`）。実際の変更点は目的変数の変換（確率差→対数比）であり、「直接 vs 引き算」ではない。
3. **循環整合性は「何も強制しない」のではなく、密なRidge版では数式上厳密に成立する**：`check_identities.py`の`check_cycle_consistency`で確認。Lasso版（ペアごとに特徴選択が変わる）でのみ崩れうる。
4. **`fit_contrastive`と`fit_ovo_logistic`のepsilon平滑化が不整合だった**：`fit_ovo_logistic`のqを$(p_{c_1}+\varepsilon)/(p_{c_1}+p_{c_2}+2\varepsilon)$に修正し、`logit(q)`が`fit_contrastive`の対数比と厳密に一致するようにした。`check_identities.py`の`check_smoothing_consistency`で確認。

**さらに「中心的な問いを精緻化してほしい」という依頼を受け**、「特定の競合クラスとの違いを説明するとき、OVRより少ない特徴で忠実に説明できるか」を研究の中心的な問いとして確定し、`docs/OVO_LIME_METHODS.md`・`docs/PROJECT_SUMMARY.md`に明記した。

## 新規実装：フェーズ12（中心的な問いの直接検証）

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

- **最優先**：提案Cの密/疎での逆転の原因調査（L1正則化パスの挙動）。
- OVR-unionの複雑さがKを超える非対称性を解消した比較（各クラスK/2個ずつなど）は未実施。
- フェーズ12で保留した仮説2（共通特徴の回避）・仮説3（競合クラス変更への追随）の検証には、共通/ペア固有/無関係特徴を明示的に作る合成データ生成器が必要で未実装。
- 前回からの持ち越し：フェーズ1〜8のfidelityのheld-out化、ソフト版Fisherのstability、`investigate_reversal.py`の統計化、実データ未検証。
- スライド（`draft_slides.pptx`）は前回・今回の訂正内容に未反映のまま。

## 次に行うこと

1. 提案Cの密/疎逆転の原因を調査する。
2. 中間発表の結論（「Cを主軸とする」）を、密/疎どちらの設定に基づくか明示する形に修正する。
3. スライドの訂正（前回から持ち越し）。
